"""STEP story (only for cutdowns): units + story map for this video, stored ones or new from Gemini.

  * if the video's folder already holds units + story map made from this exact file, use them
    (no Gemini call and no key needed)
  * otherwise build the units from the analysis (shots, transcript, speakers, people, on-screen
    text), ask Gemini ONCE for the story map, and save both

Reads:  input/<video>, and when nothing is stored: prep/*, visual/*, audio/transcript.json
Writes: prep/units.json, prep/story_map.json
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..config import PipelineConfig
from ..story import store
from ..story.story_map import request_story_map
from ..story.units import FactsAt, Unit, build_units
from ..utils.jobdir import JobDir


class StoryKeyRequired(RuntimeError):
    """No story map is stored for this video yet, and no Gemini key was given to make one."""


def lookup(job: JobDir, cfg: PipelineConfig) -> Tuple[str, Optional[List[Unit]], Optional[Dict[str, Any]]]:
    """``(fingerprint, units, record)``; units and record are None when the job folder holds nothing
    usable for this exact file (or ``cfg.refresh_story`` is set)."""
    fp = store.fingerprint(job.source)
    found = None if cfg.refresh_story else store.load(job.units_json.parent, fp)
    return (fp, found[0], found[1]) if found else (fp, None, None)


def run(job: JobDir, cfg: PipelineConfig, api_key: Optional[str] = None,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Returns ``{"fingerprint", "units", "record", "cached"}``; ``record["story_map"]`` is the map."""
    fp, units, record = lookup(job, cfg)
    if units is not None:
        log(f"  story map already stored for this video ({job.story_map_json}), no Gemini call")
        return {"fingerprint": fp, "units": units, "record": record, "cached": True}
    if not api_key:
        raise StoryKeyRequired("Gemini key required for the first cutdown of this video "
                               "(no story map is stored for it yet)")

    shots, words, facts_at = load_analysis(job, cfg)
    units = build_units(shots, words, facts_at)
    if not units:
        raise RuntimeError("Could not build any units from this video (no shots found)")
    log(f"  {len(units)} units ({sum(u.kind == 'speech' for u in units)} speech, "
        f"{sum(u.kind == 'picture' for u in units)} picture) from {len(shots)} shots and {len(words)} words")

    result = request_story_map(job.source, units, api_key, cfg.gemini_model, cfg.gemini_video_processing, log,
                               api_version=cfg.gemini_api_version)
    store.save(job.units_json.parent, fp, units, result, job.source.name)
    log(f"  stored: {job.units_json} and {job.story_map_json}")
    record = {"fingerprint": fp, "source": job.source.name, **result}
    return {"fingerprint": fp, "units": units, "record": record, "cached": False}


def load_analysis(job: JobDir, cfg: PipelineConfig) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]], FactsAt]:
    """Shots, transcript words, and a ``facts_at(start, end, shot)`` lookup, from the finished
    prep / visual / audio steps."""
    from ..models.speakers import attach_speaking_scores
    from .render import merge_shards

    transcript = job.audio_dir / "transcript.json"
    if not job.shards.exists():
        raise FileNotFoundError("The prep step has not run for this job")
    if not transcript.exists():
        raise FileNotFoundError("No transcript: the audio step (transcribe) must run before the story step")

    _, proxy, _ = job.video_infos()
    fps = proxy.fps
    sdf = pd.read_csv(job.shots).sort_values("start_sec")
    shots = list(zip(sdf.start_sec.astype(float), sdf.end_sec.astype(float)))
    shot_ids = list(sdf.shot_id.astype(int))            # position -> real shot id

    words: List[Dict[str, Any]] = []
    for seg in json.loads(transcript.read_text()):
        for w in seg.get("words_data") or []:
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append({"start": float(w["start"]), "end": float(w["end"]), "word": str(w["word"])})

    data = merge_shards(job)                            # raises if a visual shard is missing
    tracks = attach_speaking_scores(data["tracks"], data["speaker_scores"])
    people: Dict[int, int] = {}
    speaking = set()
    for frames in tracks.values():
        for f, dets in frames.items():
            persons = [d for d in dets if d.get("cname") == "person"]
            people[int(f)] = len(persons)
            if any(d.get("speaking") for d in persons):
                speaking.add(int(f))
    ocr: Dict[int, set] = {}
    for r in data["ocr"]:
        ocr.setdefault(int(r["shot_id"]), set()).add(str(r["text"]))
    speakers_known = cfg.enable_speaker_detection       # off -> "unknown", not "nobody speaks"

    def facts_at(a: float, b: float, shot_pos: int) -> Dict[str, Any]:
        fr = range(int(a * fps), max(int(a * fps) + 1, int(b * fps)))
        return {"speaking": sum(f in speaking for f in fr) / len(fr) if speakers_known else None,
                "people": max((people.get(f, 0) for f in fr), default=0),
                "ocr": sorted(ocr.get(shot_ids[shot_pos], []))}

    return shots, words, facts_at


# =========================================================================== two-call pipeline: call 1
def load_pieces_inputs(job: JobDir, cfg: PipelineConfig):
    """``(shots, words, talk spans, on-screen text by shot position)`` from the finished analysis."""
    from ..models.speakers import attach_speaking_scores
    from ..story.pieces import spans_from_frames
    from .render import merge_shards

    transcript = job.audio_dir / "transcript.json"
    if not job.shards.exists():
        raise FileNotFoundError("The prep step has not run for this job")
    if not transcript.exists():
        raise FileNotFoundError("No transcript: the audio step (transcribe) must run first")
    _, proxy, _ = job.video_infos()
    sdf = pd.read_csv(job.shots).sort_values("start_sec")
    shots = list(zip(sdf.start_sec.astype(float), sdf.end_sec.astype(float)))
    position = {int(sid): i for i, sid in enumerate(sdf.shot_id.astype(int))}

    words: List[Dict[str, Any]] = []
    for seg in json.loads(transcript.read_text()):
        for w in seg.get("words_data") or []:
            if w.get("start") is not None and w.get("end") is not None:
                words.append({"start": float(w["start"]), "end": float(w["end"]), "word": str(w["word"])})

    data = merge_shards(job)
    tracks = attach_speaking_scores(data["tracks"], data["speaker_scores"])
    talking = {int(f) for frames in tracks.values() for f, dets in frames.items()
               if any(d.get("cname") == "person" and d.get("speaking") for d in dets)}
    ocr: Dict[int, List[str]] = {}
    for r in data["ocr"]:
        texts = ocr.setdefault(position.get(int(r["shot_id"]), -1), [])
        if str(r["text"]) not in texts:
            texts.append(str(r["text"]))
    return shots, words, spans_from_frames(talking, proxy.fps), ocr


def describe(job: JobDir, cfg: PipelineConfig, api_key: Optional[str] = None,
             log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Call 1 for this video, or its stored answer. Returns ``{"audio", "video", "record", "cached"}``;
    ``record["described"]`` is Gemini's description of every unit."""
    from ..story import describe as call1
    from ..story.pieces import cut_pieces

    prep = job.units_json.parent
    fp = store.fingerprint(job.source)
    found = None if cfg.refresh_story else call1.load(prep, fp)
    if found:
        log(f"  call 1 already stored for this video ({prep / call1.CALL1_FILE}), no Gemini call")
        return {"audio": found[0], "video": found[1], "record": found[2], "cached": True}
    if not api_key:
        raise StoryKeyRequired("Gemini key required: this video has not been described yet (call 1)")
    shots, words, talk, ocr = load_pieces_inputs(job, cfg)
    audio, video = cut_pieces(shots, words, talk, ocr)
    if not video:
        raise RuntimeError("Could not cut this video into units (no shots found)")
    log(f"  {len(video)} video units (shots), {len(audio)} audio units "
        f"({sum(a.kind == 'speech' for a in audio)} with speech) from {len(words)} words")
    result = call1.request(job.source, audio, video, api_key, cfg.gemini_model, cfg.gemini_api_version, log)
    call1.save(prep, fp, audio, video, result)
    log(f"  stored: {prep / call1.PIECES_FILE} and {prep / call1.CALL1_FILE}")
    return {"audio": audio, "video": video, "record": {"fingerprint": fp, **result}, "cached": False}


# =========================================================================== two-call pipeline: call 2
def load_talk(job: JobDir) -> list:
    """Time spans in which a face is visibly talking, from the visual step."""
    from ..models.speakers import attach_speaking_scores
    from ..story.pieces import spans_from_frames
    from .render import merge_shards

    _, proxy, _ = job.video_infos()
    data = merge_shards(job)
    tracks = attach_speaking_scores(data["tracks"], data["speaker_scores"])
    talking = {int(f) for frames in tracks.values() for f, dets in frames.items()
               if any(d.get("cname") == "person" and d.get("speaking") for d in dets)}
    return spans_from_frames(talking, proxy.fps)


def plan_edits(job: JobDir, cfg: PipelineConfig, described: Dict[str, Any], api_key: Optional[str] = None,
               log: Callable[[str], None] = print) -> Dict[str, Dict[str, Any]]:
    """Call 2 for every requested length and mode (stored edits are reused), turned into exact pieces.

    ``described`` is what ``describe`` returned. Returns cuts in the shape ``render.run_cuts`` takes,
    named ``<seconds>s_<mode>_gemini``.
    """
    from ..story import edit_plan as call2

    audio, video, desc = described["audio"], described["video"], described["record"]["described"]
    prep = job.units_json.parent
    fp = described["record"]["fingerprint"]
    stored = {} if cfg.refresh_edits else call2.load(prep, fp)
    source = video[-1].end
    talk = load_talk(job)
    usable = [v.id for v in video if call2.cover_room(v, talk)]
    if "cutaway" in cfg.cut_modes:
        log(f"  shots usable as a cover (a stretch without a talking face): {' '.join(usable) or 'NONE'}")
    for mode in cfg.cut_modes:
        missing = [d for d in cfg.cut_durations if d < source and call2.name(d, mode) not in stored]
        if not missing:
            continue
        if not api_key:
            log(f"  ✗ {mode}: no stored edit for {', '.join(f'{d:g}s' for d in missing)} and no Gemini key given")
            continue
        try:
            stored.update(call2.request(audio, video, desc, missing, mode, api_key, cfg.gemini_model,
                                        cfg.gemini_api_version, log, talk))
            call2.save(prep, fp, stored)
        except call2.GeminiError as e:
            log(f"  ✗ {mode}: {e}")
    cuts: Dict[str, Dict[str, Any]] = {}
    for d in cfg.cut_durations:
        for mode in cfg.cut_modes:
            key = call2.name(d, mode)
            if d >= source:
                log(f"  {key}: skipped, the video is only {source:.1f}s long")
            elif key in stored:
                cuts[key + "_gemini"] = call2.build_cut(audio, video, desc, stored[key], talk)
    return cuts


# =========================================================================== optional: framing check
def framing_check(job: JobDir, cfg: PipelineConfig, api_key: Optional[str] = None,
                  log: Callable[[str], None] = print) -> list:
    """Gemini's corrections to our focal points, as ``[(start_s, end_s, x percent)]`` (``[]`` if none, or if
    there is no stored answer for this framing and no key to ask with). Separate from call 1: it needs only
    the shots and our focal points, no transcript, and stores its own file (prep/framing_check.json)."""
    from ..story import framing_check as check
    from ..story.pieces import build_video_units
    from . import render

    sdf = pd.read_csv(job.shots).sort_values("start_sec")
    video = build_video_units(list(zip(sdf.start_sec.astype(float), sdf.end_sec.astype(float))))
    centre, src, proxy = render.focus_line(job, cfg)
    line = check.signature([centre[f] for f in sorted(centre)])
    prep = job.units_json.parent
    fp = store.fingerprint(job.source)
    stored = None if cfg.refresh_story else check.load(prep, fp, line)
    if stored is None:
        if not api_key:
            log("  framing check skipped: no stored answer for this framing and no Gemini key given")
            return []
        preview = render.focus_preview(job, centre, src, proxy, [(v.start, v.end, v.id) for v in video], log)
        stored = check.request(preview, video, api_key, cfg.gemini_model, cfg.gemini_api_version, log)
        check.save(prep, fp, line, stored)
    else:
        log(f"  framing check already stored for this framing ({prep / check.FRAMING_FILE}), no Gemini call")
    log(check.report(video, stored["shots"]))
    return check.focus_fixes(video, stored["shots"])
