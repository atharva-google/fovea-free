"""Call 2: Gemini builds the edits from call 1's stored descriptions. Text only: no video is sent.

One request per mode, covering every requested length:

    synced    input = AV segments (a sound with the picture it was recorded with); Gemini returns
              which segments to keep, in order
    cutaway   input = audio units and video units (shots) separately; Gemini returns sound + picture
              pairs, so a line can be heard over a different shot

Gemini only names ids. Code turns its answer into exact pieces, and enforces the mechanics:
unknown ids are dropped, a sound plays once, a shot used to cover a line is not also shown on its
own, a cover never repeats footage, and a cover avoids the parts of a shot where a face is talking.
If a kept piece ``needs`` one that Gemini cut, the cut one is put back (the story comes before the
length). Beyond that, whether the story holds is Gemini's job, and the viewer's to judge.

Stored in the video's own folder: prep/call2.json (one entry per length and mode; reused as is).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .describe import REQUEST_TIMEOUT_S, GeminiError
from .pieces import AudioUnit, VideoUnit
from .select import Piece, merge_pieces

CALL2_VERSION = 2
CALL2_FILE = "call2.json"
MIN_COVER = 0.8      # a cover picture shorter than this is not worth a cut
MODES = ("synced", "cutaway")

Span = Tuple[float, float]

_COMMON = """You are a senior video editor cutting an ad down to shorter versions.
You cannot see the video. Another editor watched it and logged every piece of it for you (below).

THE AD: {summary}
Continuous background music: {music}
Full length: {source:.1f}s. Make one edit for EACH of these lengths: {targets}.

How to edit:
- The story comes first. Every edit must still work for someone who never saw the full ad: keep what the payoff
  depends on, keep the product / brand moment and the closing card or offer if there is one.
- Aim for the requested length. The pieces below have exact lengths: add them up. Slightly over is fine if the
  story needs it; an edit must always be clearly shorter than the full ad.
- Each length is its own edit. A shorter one has to make harder choices than a longer one, not be the same edit.
- Where to find time, in this order: shots the story does not need; silences, pauses and exclamations such as
  (aaa) or (silence); the beginning or end of a shot; repeated or decorative lines.
- Do not break a sentence: pieces of speech that form one sentence are kept or cut together.
- "needs" tells you what a piece makes no sense without. If you keep a piece, keep what it needs.
- Keeping the beginning and the end of a shot while cutting its middle makes the picture jump. Avoid it unless
  what you remove is only a short pause.
- Keep the original order unless moving one strong moment to the very start clearly makes a better opening.
- Before you answer, read each edit from start to finish using ONLY the pieces you kept. Nothing kept may react
  to, refer to, or pay off something you cut. Check the ending hardest: if you cut a gag, also cut (or do not
  choose) the pieces that call back to it, and end on something the kept story has earned.
- Refer to pieces ONLY by their id. Never write timestamps.
"""

SYNCED_PROMPT = _COMMON + """
This is the SYNCED edit: every sound plays with the picture it was recorded with.
The ad is split into SEGMENTS (S01, S02, ...): a few spoken words, or the sound between words, together with
the shot(s) they were recorded over. Return, for each length, the segments to keep in playing order.

SHOTS (what is seen)
{shots}

SEGMENTS (what is heard, and over which shot)
{segments}
"""

CUTAWAY_PROMPT = _COMMON + """
This is the CUTAWAY edit: a sound may be heard over a DIFFERENT shot than the one it was recorded with.
You get the sound and the picture as separate pieces:
  AUDIO UNITS (A01, ...)  a few spoken words, or the sound between words; "over" is the shot it was recorded with
  VIDEO UNITS (V01, ...)  one per shot
Return, for each length, a list of segments in playing order. Each segment is one audio unit AND the video unit
that is SEEN while it plays. Always give both. To show the picture a sound was recorded with, name the shot in
its "over"; to cut away, name a different shot.

A cutaway edit MUST cut away. An answer in which every sound plays over its own shot is wrong: that is just the
synced edit. At least two segments in every edit (more in a longer edit) must show a different shot than the one
the sound was recorded over. Do not start from a synced edit; build it this way:
  1. Choose the SOUNDS that tell the story (the lines that must be heard).
  2. Choose the SHOTS that must be seen (the product, the reveal, the result, the closing card).
  3. Lay those shots over the chosen sounds. Good sounds to cover: voiceover and narration (nobody is seen saying
     it), a line by someone we have already seen, music or silence. Good shots to cover with: the product, an
     object, a reaction, a result: shots with "visibly talking" near 0%.
  4. Drop the audio units recorded over every shot you used as a cover, so that shot is seen once.

How to use cutaways:
- This is how time is saved without losing anything: instead of playing a line and THEN showing a shot, show the
  shot WHILE the line plays (name the gift while we see the gift). The covering shot then needs no time of its own.
- A shot you use to cover a sound must not also appear on its own (do not also list the audio units recorded over it).
- A covering shot may cover several audio units; it keeps playing from where it left off.
- The picture is shown for exactly as long as the sound lasts. Check the lengths: a 1.2s shot cannot cover 3s.
- Never cover speech with a shot in which someone else is visibly talking ("visibly talking" near 100%).
- A sound that belongs to what is seen, like (door slams), stays with its own picture.
- Do not cover a line when seeing the person say it is the point (a reaction, a punchline delivered to camera).

VIDEO UNITS
{video}

AUDIO UNITS
{audio}
"""


def name(target: float, mode: str) -> str:
    return f"{target:g}s_{mode}"


def _sid(aid: str) -> str:
    return "S" + aid[1:]


def _needs(ids: Sequence[str], as_segments: bool) -> str:
    ids = [_sid(n) if as_segments and n.startswith("A") else n for n in ids]
    return f" | needs {' '.join(ids)}" if ids else ""


def synced_prompt(audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any],
                  targets: Sequence[float]) -> str:
    dv = {u["id"]: u for u in described["video_units"]}
    da = {u["id"]: u for u in described["audio_units"]}
    shots = "\n".join(f"{v.id} | {v.dur:.2f}s | shows: {dv[v.id]['shows']}{_needs(dv[v.id]['needs'], True)}"
                      for v in video)
    segs = "\n".join(f"{_sid(a.id)} | {a.dur:.2f}s | over {' '.join(a.over)} | heard: {da[a.id]['heard']}"
                     f"{_needs(da[a.id]['needs'], True)}" for a in audio)
    return _fill(SYNCED_PROMPT, described, video, targets, shots=shots, segments=segs)


def cutaway_prompt(audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any],
                   targets: Sequence[float]) -> str:
    dv = {u["id"]: u for u in described["video_units"]}
    da = {u["id"]: u for u in described["audio_units"]}
    vid = "\n".join(f"{v.id} | {v.dur:.2f}s | visibly talking: {v.talking:.0%} | shows: {dv[v.id]['shows']} | "
                    f"focus: {dv[v.id]['focus']}{_needs(dv[v.id]['needs'], False)}" for v in video)
    aud = "\n".join(f"{a.id} | {a.dur:.2f}s | over {' '.join(a.over)} | heard: {da[a.id]['heard']}"
                    f"{_needs(da[a.id]['needs'], False)}" for a in audio)
    return _fill(CUTAWAY_PROMPT, described, video, targets, video=vid, audio=aud)


def _fill(template: str, described: Dict[str, Any], all_shots: List[VideoUnit], targets: Sequence[float], **tables) -> str:
    music = {"bed": "yes", "only": "yes"}.get(described.get("music"), "no")
    out = template
    for key, value in {"summary": described.get("summary", ""), "music": music,
                       "source": f"{all_shots[-1].end:.1f}", "targets": ", ".join(f"{t:g}s" for t in targets),
                       **tables}.items():
        out = out.replace("{" + key + "}", str(value)).replace("{" + key + ":.1f}", str(value))
    return out


# =========================================================================== the Gemini call
def request(audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any], targets: Sequence[float],
            mode: str, api_key: str, model: str = "gemini-3.8-flash", api_version: str = "v1beta",
            log: Callable[[str], None] = print) -> Dict[str, Dict[str, Any]]:
    """One text-only request for ``mode``. Returns ``{"15s_synced": {"target_s", "mode", "segments",
    "story", "model"}}`` with ``segments`` as ``[{"audio", "video"}]`` in both modes."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not api_key:
        raise GeminiError("A Gemini API key is required for call 2")
    try:
        from google import genai
        from google.genai import types

        from .schemas import CutawayEdits, SyncedEdits
    except ImportError as e:
        raise GeminiError(f"google-genai (and pydantic) must be installed: pip install -U google-genai ({e})") from e

    text = (synced_prompt if mode == "synced" else cutaway_prompt)(audio, video, described, targets)
    schema = SyncedEdits if mode == "synced" else CutawayEdits
    http = {"timeout": REQUEST_TIMEOUT_S * 1000}
    if api_version:
        http["api_version"] = api_version
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(**http))
    log(f"  call 2 ({mode}): asking {model} for {', '.join(f'{t:g}s' for t in targets)}")
    try:
        t0 = time.time()
        resp = client.models.generate_content(
            model=model, contents=[text],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema))
        log(f"  answered in {time.time() - t0:.1f}s")
        raw = json.loads(resp.text or "")
    except ValueError:
        raise GeminiError("Gemini's answer was not valid JSON (it may have been cut off); run the command again")
    except Exception as e:  # noqa: BLE001 - never leak the key
        if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower():
            raise GeminiError(f"Gemini did not answer within {REQUEST_TIMEOUT_S}s; run the command again") from None
        raise GeminiError(f"Gemini call failed: {type(e).__name__}: {e}".replace(api_key, "***")) from None
    return parse(raw, targets, mode, model)


def parse(raw: Dict[str, Any], targets: Sequence[float], mode: str, model: str = "") -> Dict[str, Dict[str, Any]]:
    """Match Gemini's edits to the requested lengths (by ``target_s``, else by position)."""
    edits = [e for e in raw.get("edits") or [] if isinstance(e, dict)]
    out: Dict[str, Dict[str, Any]] = {}
    used = set()
    for i, t in enumerate(targets):
        pick = next((k for k, e in enumerate(edits) if k not in used and _close(e.get("target_s"), t)), None)
        if pick is None and i < len(edits) and i not in used:
            pick = i
        if pick is None:
            continue
        used.add(pick)
        e = edits[pick]
        if mode == "synced":
            segs = [{"audio": "A" + str(s).strip()[1:], "video": ""} for s in e.get("segments") or []]
        else:
            segs = [{"audio": str(s.get("audio") or "").strip(), "video": str(s.get("video") or "").strip()}
                    for s in e.get("segments") or [] if isinstance(s, dict)]
        out[name(t, mode)] = {"target_s": float(t), "mode": mode, "segments": segs,
                              "story": str(e.get("story") or ""), "model": model}
    return out


def _close(value: Any, target: float) -> bool:
    try:
        return abs(float(value) - target) < 0.5
    except (TypeError, ValueError):
        return False


# =========================================================================== storage
def load(folder: str | Path, fp: str) -> Dict[str, Dict[str, Any]]:
    """Stored edits for this exact file (``{}`` if none)."""
    path = Path(folder) / CALL2_FILE
    try:
        d = json.loads(path.read_text())
        if d.get("fingerprint") == fp and d.get("call2_version") == CALL2_VERSION:
            return dict(d.get("edits") or {})
    except (OSError, ValueError):
        pass
    return {}


def save(folder: str | Path, fp: str, edits: Dict[str, Dict[str, Any]]) -> None:
    path = Path(folder) / CALL2_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"fingerprint": fp, "call2_version": CALL2_VERSION, "edits": edits},
                              indent=1, ensure_ascii=False))
    tmp.replace(path)


# =========================================================================== edit -> exact pieces
def quiet_spans(v: VideoUnit, talk: Sequence[Span]) -> List[Span]:
    """The stretches of a shot in which no face is visibly talking."""
    out, t = [], v.start
    for x, y in sorted((max(v.start, float(x)), min(v.end, float(y))) for x, y in talk):
        if y <= x:
            continue
        if x > t:
            out.append((t, x))
        t = max(t, y)
    if t < v.end:
        out.append((t, v.end))
    return out


def build_cut(audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any], edit: Dict[str, Any],
              talk: Sequence[Span] = ()) -> Dict[str, Any]:
    """``{"pieces": [...], "stats": {...}}`` for one stored edit, in the shape the renderer takes."""
    A, V = {a.id: a for a in audio}, {v.id: v for v in video}
    mode, target = edit["mode"], float(edit["target_s"])
    notes: List[str] = []
    segs: List[List[str]] = []
    seen = set()
    for s in edit.get("segments") or []:
        aid, vid = s.get("audio", ""), s.get("video", "") if mode == "cutaway" else ""
        if aid not in A:
            notes.append(f"unknown piece {aid or '(empty)'!r} ignored")
            continue
        if aid in seen:
            notes.append(f"{aid} listed twice: played once")
            continue
        seen.add(aid)
        if vid and vid not in V:
            notes.append(f"{aid}: unknown picture {vid!r}, its own picture is used")
            vid = ""
        if vid in A[aid].over:
            vid = ""
        segs.append([aid, vid])

    quiet = {v.id: quiet_spans(v, talk) for v in video}
    dv = {u["id"]: u for u in described["video_units"]}
    da = {u["id"]: u for u in described["audio_units"]}

    def put_back_needed() -> bool:
        """The story first: if a kept piece needs one that was cut, the cut one goes back in. True if anything
        was added (one need per call)."""
        kept = [aid for aid, _ in segs]
        shown = {v for aid, vid in segs for v in ([vid] if vid else A[aid].over)}
        for uid in [*kept, *sorted(shown)]:
            for n in (da.get(uid) or dv.get(uid))["needs"]:
                if n in kept or n in shown:
                    continue
                add = [n] if n in A else [a.id for a in audio if n in a.over and a.id not in kept]
                if not add:
                    continue
                for new in add:
                    at = next((i for i, (aid, _) in enumerate(segs) if A[aid].start > A[new].start), len(segs))
                    segs.insert(at, [new, ""])
                notes.append(f"{uid} needs {n}, which was cut: put back ({' '.join(add)})")
                return True
        return False

    def allocate() -> List[Optional[Tuple[float, float, float]]]:
        """Per segment with a cover: (picture start, cover length, seconds of its own picture first)."""
        cursor = {v.id: v.start for v in video}
        out: List[Optional[Tuple[float, float, float]]] = []
        for aid, vid in segs:
            if not vid:
                out.append(None)
                continue
            need, v = A[aid].dur, V[vid]
            win = None
            for least in (need, min(need, MIN_COVER)):   # first a quiet stretch that fits the whole sound
                for a, b in quiet[vid]:
                    a = max(a, cursor[vid])
                    if b - a >= least - 1e-6:
                        win = (a, min(need, b - a))
                        break
                if win:
                    break
            if win is None:
                out.append(None)
                continue
            start, length = win
            if 0 < need - length < 0.3:          # don't flash the speaker for a few frames: let the cover run on
                length = min(need, v.end - start)
            cursor[vid] = start + length
            out.append((start, length, max(0.0, need - length)))
        return out

    while True:                                   # undo covers that cannot work, until nothing changes
        own_shots = {v for aid, vid in segs if not vid for v in A[aid].over}
        plan = allocate()
        bad = None
        for i, (aid, vid) in enumerate(segs):
            if vid and vid in own_shots:
                bad = (i, f"{vid} is also shown on its own, so {aid} keeps its own picture")
            elif vid and plan[i] is None:
                bad = (i, f"{vid} has no {min(A[aid].dur, MIN_COVER):.1f}s left without a talking face to cover "
                          f"{aid}, which keeps its own picture")
            if bad:
                break
        if bad is not None:
            segs[bad[0]][1] = ""
            notes.append(bad[1])
        elif not put_back_needed():
            break

    if mode == "cutaway" and segs and not any(vid for _, vid in segs):
        notes.append("NO CUTAWAYS: every sound plays over its own picture, so this is a synced edit")
    pieces: List[Piece] = []
    for (aid, vid), win in zip(segs, plan):
        a = A[aid]
        if not vid:
            pieces.append(Piece(a.start, a.start, a.dur, aid, da[aid]["heard"][:60]))
            continue
        start, length, own = win
        if own > 1e-3:
            pieces.append(Piece(a.start, a.start, own, aid, "its own picture first"))
        pieces.append(Piece(start, a.start + own, length, f"{aid}<-{vid}", f"over: {dv[vid]['shows']}"[:80]))
    pieces = merge_pieces(pieces, bridge_gap=1e-3)

    total = round(sum(p.length for p in pieces), 2)
    source = video[-1].end
    ok = bool(pieces) and total < 0.95 * source
    if not pieces:
        notes.append("the edit is empty")
    elif not ok:
        notes.append(f"not meaningfully shorter than the source ({total:.1f}s of {source:.1f}s)")
    elif abs(total - target) > max(1.0, 0.15 * target):
        notes.append(f"{abs(total - target):.1f}s {'over' if total > target else 'under'} the {target:g}s target")
    return {"pieces": [p.to_dict() for p in pieces],
            "stats": {"target": target, "mode": mode, "total": total, "units": [aid for aid, _ in segs],
                      "covers": {aid: vid for aid, vid in segs if vid}, "notes": notes, "ok": ok,
                      "story": edit.get("story", "")}}


def report(cuts: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    for cut_name, cut in cuts.items():
        st = cut["stats"]
        flag = "" if st["ok"] or not cut["pieces"] else "  NOT USABLE"
        lines.append(f"=== {cut_name}: {st['total']:g}s in {len(cut['pieces'])} piece(s){flag}")
        if st.get("story"):
            lines.append(f"   Gemini: {st['story']}")
        for n in st["notes"]:
            lines.append(f"   note: {n}")
        for p in cut["pieces"]:
            where = f"{p['video_start']:6.2f}s"
            if abs(p["video_start"] - p["audio_start"]) > 1e-6:
                where += f" (sound from {p['audio_start']:.2f}s)"
            lines.append(f"   {p['length']:5.2f}s @ {where}  {p['units']}  {p['note']}")
    return "\n".join(lines)
