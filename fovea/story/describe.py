"""Call 1: Gemini watches the video ONCE and describes every unit. Stored; never repeated for a file.

For the video: a cast list (who appears, how they look, their names if known).
For each video unit (a shot): what is seen, which cast members are in it, the setting, the objects
that matter, its on-screen text, and which earlier units it depends on. Call 1 is about the story and the units
only; checking the framing is a separate, optional call (``framing_check.py``).
For each audio unit: what is really heard (the transcript corrected), who says it, and which earlier
units it depends on (a punchline needs its setup).

One request, with an output schema (fovea/story/schemas.py), standard video processing.

Stored in the video's own folder:
    prep/pieces.json   the units cut by code (times, transcript, detector facts)
    prep/call1.json    Gemini's descriptions of them
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .pieces import PIECES_VERSION, AudioUnit, VideoUnit

CALL1_VERSION = 7           # bump when the prompt or schema changes in a way that makes stored answers stale
REQUEST_TIMEOUT_S = 300
PROCESSING_TIMEOUT_S = 180
PIECES_FILE, CALL1_FILE = "pieces.json", "call1.json"

PROMPT = """You are a senior video editor logging the footage of an ad before it is cut down.
Watch and listen to the whole video carefully. A second editor will later build shorter versions of this ad
from your log WITHOUT seeing the video, so everything they need to know has to be in it.

The video has already been cut into units with exact times (listed below):
  VIDEO UNITS  one per shot (V01, V02, ...)
  AUDIO UNITS  a few spoken words, or the sound between words (A01, A02, ...). "over" names the shot(s)
               the sound was recorded over.
Refer to units ONLY by their id. Never write timestamps.

The measurements next to each unit come from analysis software and can be wrong:
  visibly talking   share of the shot in which a face is seen talking
  detected text     on-screen text read by OCR (often garbled)
  rough transcript  speech-to-text; it may have wrong words, miss sounds, or invent words
Trust what you see and hear over these.

For EVERY video unit give:
  shows    what is seen, in one sentence: who, doing what. People who are not in the cast list (a crowd,
           passers-by) are mentioned here.
  cast     the ids of the cast members (see "cast" below) recognisably visible at ANY point in this shot, even
           briefly, partly or from behind
  setting  where it is, as a short label ("living room", "beach", "studio, white background"). Use exactly the
           same wording for every shot in the same place.
  objects  things that matter to the story or that someone would name: the product, props that are handled,
           vehicles, animals. A few words each, and the same wording in every shot they appear in.
  text     readable on-screen text that belongs to THIS shot (a price, a caption, a sign, end-card wording),
           corrected. Leave out a logo or disclaimer that sits in the same place throughout the video.
  needs    earlier units this shot makes NO SENSE without (a reaction needs what it reacts to; a reveal needs
           its setup). A shot does NOT need another one just because it comes after it, shows the same scene or
           continues the same action: ask "if the earlier unit were cut, would a viewer be confused by this
           shot?" Most shots need nothing. Only the DIRECT ones; do not list the whole chain.
           Look hardest at the END of the video: a closing shot that calls back to an earlier moment (a joke, a
           claim, a demonstration), shows its result, or repeats it, needs that moment. If the earlier moment
           were cut, would this shot still make sense? If not, it needs it.

For EVERY audio unit give:
  heard    what is REALLY heard. Write speech out exactly, in its own language and script, correcting the rough
           transcript. Put any other sound in round brackets: (aaa), (laughs), (door slams), (music), (silence).
  needs    earlier units this sound makes NO SENSE without (a punchline needs its setup line; "look at this"
           needs the shot it points at; a line that refers back to an earlier moment needs that moment).
           A sound does NOT need the one before it just because it follows it, and the parts of one sentence
           do not need each other. Most sounds need nothing. Only the DIRECT ones.
  speaker  who says it: a cast id; "voiceover" for a narrator who is not in the cast; "other" for any other
           unseen or unlisted voice; empty when nothing is spoken

Also give:
  cast     everyone who speaks or is clearly the subject of at least one shot, as P1, P2, ... in order of first
           appearance. NOT background people or crowds. The same person is ONE entry however often they appear,
           also if their clothes or the lighting change. For each:
             real_name    their real name ONLY if this is a well-known person you recognise with confidence;
                          otherwise empty. Never guess.
             ad_name      the name the video itself gives them (spoken or on screen); otherwise empty
             description  how they LOOK, so they can be told apart: apparent age and gender, hair, build,
                          clothing. Nothing about what they do in the story.
  summary  one or two sentences: the story of the ad
  music    "bed" if continuous music runs under the ad, "none" if not, "only" if the file is nothing but music

{table}
"""


def units_table(audio: List[AudioUnit], video: List[VideoUnit]) -> str:
    lines = ["VIDEO UNITS"]
    for v in video:
        text = f" | detected text: {'; '.join(v.ocr)}" if v.ocr else ""
        lines.append(f"{v.id} | {v.start:.2f}-{v.end:.2f}s ({v.dur:.2f}s) | visibly talking: {v.talking:.0%}{text}")
    lines.append("")
    lines.append("AUDIO UNITS")
    for a in audio:
        said = f'rough transcript: "{a.text}"' if a.kind == "speech" else "(no words transcribed)"
        lines.append(f"{a.id} | {a.start:.2f}-{a.end:.2f}s ({a.dur:.2f}s) | over {' '.join(a.over)} | {said}")
    return "\n".join(lines)


def prompt(audio: List[AudioUnit], video: List[VideoUnit]) -> str:
    return PROMPT.replace("{table}", units_table(audio, video))


# =========================================================================== cleaning the answer
def clean(raw: Dict[str, Any], audio: List[AudioUnit], video: List[VideoUnit]) -> Tuple[Dict[str, Any], List[str]]:
    """Keep only known ids, fill units Gemini forgot, and keep only ``needs`` that point at EARLIER units."""
    issues: List[str] = []
    start = {u.id: u.start for u in [*audio, *video]}

    def needs(uid: str, value: Any) -> List[str]:
        out = []
        for n in value or []:
            n = str(n).strip()
            if n not in start:
                issues.append(f"{uid} needs unknown unit {n!r}: ignored")
            elif n == uid or start[n] > start[uid] + 1e-6:
                issues.append(f"{uid} needs {n}, which is not earlier: ignored")
            elif n not in out:
                out.append(n)
        return out

    def by_id(items: Any) -> Dict[str, Dict[str, Any]]:
        return {str(x.get("id", "")).strip(): x for x in items or [] if isinstance(x, dict)}

    def words(value: Any) -> List[str]:
        out = []
        for x in value or [] if isinstance(value, (list, tuple)) else []:
            x = str(x).strip()
            if x and x not in out:
                out.append(x)
        return out

    cast = []
    for c in raw.get("cast") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        if not cid or cid in {x["id"] for x in cast}:
            issues.append(f"cast entry with a missing or repeated id {cid!r} ignored")
            continue
        cast.append({"id": cid, "real_name": str(c.get("real_name") or "").strip(),
                     "ad_name": str(c.get("ad_name") or "").strip(), "description": str(c.get("description") or "").strip()})
    cast_ids = {c["id"] for c in cast}

    got_v, got_a = by_id(raw.get("video_units")), by_id(raw.get("audio_units"))
    for uid in [*got_v, *got_a]:
        if uid not in start:
            issues.append(f"unknown unit {uid!r} ignored")
    video_out, audio_out = [], []
    for v in video:
        g = got_v.get(v.id)
        if g is None:
            issues.append(f"{v.id} missing from the answer")
            g = {}
        seen = words(g.get("cast"))
        for c in seen:
            if c not in cast_ids:
                issues.append(f"{v.id} lists {c!r}, who is not in the cast: ignored")
        video_out.append({"id": v.id, "shows": str(g.get("shows") or ""), "cast": [c for c in seen if c in cast_ids],
                          "setting": str(g.get("setting") or "").strip(), "objects": words(g.get("objects")),
                          "text": words(g.get("text")), "needs": needs(v.id, g.get("needs"))})
    for a in audio:
        g = got_a.get(a.id)
        if g is None:
            issues.append(f"{a.id} missing from the answer")
            g = {}
        heard = str(g.get("heard") or "") or (a.text if a.kind == "speech" else "(silence)")
        speaker = str(g.get("speaker") or "").strip()
        if speaker and speaker not in cast_ids and speaker not in ("voiceover", "other"):
            issues.append(f"{a.id} is said by {speaker!r}, who is not in the cast: recorded as 'other'")
            speaker = "other"
        audio_out.append({"id": a.id, "heard": heard, "speaker": speaker, "needs": needs(a.id, g.get("needs"))})
    music = str(raw.get("music") or "none")
    return {"summary": str(raw.get("summary") or ""), "music": music if music in ("none", "bed", "only") else "none",
            "cast": cast, "video_units": video_out, "audio_units": audio_out}, issues


def cast_name(c: Dict[str, Any]) -> str:
    """``P1 (Jigyasa / Alia Bhatt)``: the id with whatever names are known."""
    names = " / ".join(n for n in (c.get("ad_name"), c.get("real_name")) if n)
    return f"{c['id']} ({names})" if names else c["id"]


# =========================================================================== the Gemini call
class GeminiError(RuntimeError):
    """A Gemini call failed. The message never contains the API key."""


def ask_about_video(video_path: str | Path, text: str, schema: str, api_key: str, model: str, api_version: str,
                    log: Callable[[str], None], what: str) -> Tuple[Dict[str, Any], float]:
    """Upload a video, ask one question about it with an output schema (a class name in ``schemas.py``),
    and return ``(the answer as a dict, seconds it took)``. The upload is deleted afterwards."""
    if not api_key:
        raise GeminiError(f"A Gemini API key is required for {what}")
    try:
        from google import genai
        from google.genai import types

        from . import schemas
    except ImportError as e:
        raise GeminiError(f"google-genai (and pydantic) must be installed: pip install -U google-genai ({e})") from e

    http = {"timeout": REQUEST_TIMEOUT_S * 1000}
    if api_version:
        http["api_version"] = api_version
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(**http))
    uploaded = None
    try:
        log(f"  uploading {Path(video_path).name} to Gemini")
        uploaded = client.files.upload(file=str(video_path))
        log("  uploaded; waiting for Gemini to make the video usable")
        waited = 0.0
        while uploaded.state and uploaded.state.name == "PROCESSING":
            if waited > PROCESSING_TIMEOUT_S:
                raise GeminiError(f"Gemini is still processing the uploaded video after {PROCESSING_TIMEOUT_S}s; "
                                  "run the command again")
            time.sleep(2)
            waited += 2
            uploaded = client.files.get(name=uploaded.name)
        if not uploaded.state or uploaded.state.name != "ACTIVE":
            raise GeminiError(f"Gemini could not process the video (state: {getattr(uploaded.state, 'name', '?')})")

        part = types.Part(file_data=types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type))
        log(f"  {what}: asking {model} (gives up after {REQUEST_TIMEOUT_S}s)")
        t0 = time.time()
        resp = client.models.generate_content(
            model=model, contents=[part, text],
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=getattr(schemas, schema)))
        seconds = round(time.time() - t0, 1)
        usage = getattr(resp, "usage_metadata", None)
        log(f"  answered in {seconds}s | tokens in={getattr(usage, 'prompt_token_count', None)} "
            f"out={getattr(usage, 'candidates_token_count', None)}")
        try:
            return json.loads(resp.text or ""), seconds
        except ValueError:
            raise GeminiError("Gemini's answer was not valid JSON (it may have been cut off); run the command again")
    except GeminiError:
        raise
    except Exception as e:  # noqa: BLE001 - SDK errors can echo request details; never leak the key
        if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower():
            raise GeminiError(f"Gemini did not answer within {REQUEST_TIMEOUT_S}s; run the command again") from None
        raise GeminiError(f"Gemini call failed: {type(e).__name__}: {e}".replace(api_key, "***")) from None
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:  # noqa: BLE001 - uploaded files expire on their own
                pass


def request(video_path: str | Path, audio: List[AudioUnit], video: List[VideoUnit], api_key: str,
            model: str = "gemini-3.8-flash", api_version: str = "v1beta",
            log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Returns ``{"described", "issues", "model", "seconds"}``; ``described`` is the cleaned answer."""
    log(f"  call 1: {len(video)} video and {len(audio)} audio units to describe")
    raw, seconds = ask_about_video(video_path, prompt(audio, video), "Units", api_key, model, api_version, log, "call 1")
    described, issues = clean(raw, audio, video)
    for issue in issues[:10]:
        log(f"  call 1: {issue}")
    return {"described": described, "issues": issues, "model": model, "seconds": seconds}


# =========================================================================== storage
def save(folder: str | Path, fp: str, audio: List[AudioUnit], video: List[VideoUnit], result: Dict[str, Any]) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / PIECES_FILE, {"fingerprint": fp, "pieces_version": PIECES_VERSION,
                                  "video_units": [v.to_dict() for v in video],
                                  "audio_units": [a.to_dict() for a in audio]})
    _write(folder / CALL1_FILE, {"fingerprint": fp, "call1_version": CALL1_VERSION, **result})   # last = "done"


def load(folder: str | Path, fp: str) -> Optional[Tuple[List[AudioUnit], List[VideoUnit], Dict[str, Any]]]:
    """``(audio, video, record)`` if the folder holds call 1's answer for this exact file;
    ``record["described"]`` is the answer."""
    pieces, call1 = Path(folder) / PIECES_FILE, Path(folder) / CALL1_FILE
    if not (pieces.is_file() and call1.is_file()):
        return None
    try:
        p, record = json.loads(pieces.read_text()), json.loads(call1.read_text())
        audio = [AudioUnit.from_dict(d) for d in p["audio_units"]]
        video = [VideoUnit.from_dict(d) for d in p["video_units"]]
        described = record["described"]
        ids = [u["id"] for u in described["video_units"]] + [u["id"] for u in described["audio_units"]]
    except (ValueError, KeyError, TypeError):
        return None
    if p.get("fingerprint") != fp or record.get("fingerprint") != fp:
        return None
    if p.get("pieces_version") != PIECES_VERSION or record.get("call1_version") != CALL1_VERSION:
        return None
    if ids != [v.id for v in video] + [a.id for a in audio]:
        return None
    return audio, video, record


def _write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


# =========================================================================== for people
def report(audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any]) -> str:
    dv = {u["id"]: u for u in described["video_units"]}
    da = {u["id"]: u for u in described["audio_units"]}
    lines = [f"SUMMARY: {described['summary']}", f"MUSIC: {described['music']}", "", "CAST"]
    lines += [f"  {cast_name(c)}: {c['description']}" for c in described.get("cast") or []] or ["  (nobody)"]
    lines += ["", "VIDEO UNITS (one per shot)"]
    for v in video:
        d = dv[v.id]
        need = f"  needs {' '.join(d['needs'])}" if d["needs"] else ""
        lines.append(f"  {v.id}  {v.start:6.2f}-{v.end:6.2f}s {v.dur:5.2f}s  talking {v.talking:4.0%}{need}")
        lines.append(f"        shows: {d['shows']}")
        lines.append(f"        cast: {' '.join(d.get('cast') or []) or '-'}  |  setting: {d.get('setting') or '-'}")
        if d.get("objects"):
            lines.append(f"        objects: {'; '.join(d['objects'])}")
        if d.get("text"):
            lines.append(f"        text: {'; '.join(d['text'])}")
    lines += ["", "AUDIO UNITS"]
    for a in audio:
        d = da[a.id]
        need = f"  needs {' '.join(d['needs'])}" if d["needs"] else ""
        who = f"{d['speaker']}: " if d.get("speaker") else ""
        lines.append(f"  {a.id}  {a.start:6.2f}-{a.end:6.2f}s {a.dur:5.2f}s  over {' '.join(a.over):<8} "
                     f"heard: {who}{d['heard']}{need}")
        if a.kind == "speech" and d["heard"].strip() != a.text.strip():
            lines.append(f"        (rough transcript was: {a.text})")
    return "\n".join(lines)
