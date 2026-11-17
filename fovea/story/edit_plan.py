"""Call 2: Gemini builds the edits from call 1's stored descriptions. Text only: no video is sent.

One request per mode, covering every requested length:

    synced    input = AV segments (a sound with the picture it was recorded with); Gemini returns
              which segments to keep, in order
    cutaway   input = audio units and video units (shots) separately; Gemini returns sound + picture
              pairs, so a line can be heard over a different shot

Gemini only names ids. Code turns its answer into exact pieces, and enforces the mechanics:
unknown ids are dropped, a sound plays once, no footage is ever shown twice (a shot may be cut back
to, but a cover only takes footage of it that is not shown elsewhere), and a cover avoids the parts
of a shot where a face is talking. Whether the story holds is Gemini's job, and the viewer's to
judge: a kept piece whose ``needs`` were cut is reported, not changed.

Stored in the video's own folder: prep/call2.json (one entry per length and mode; reused as is).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .describe import REQUEST_TIMEOUT_S, GeminiError, cast_name
from .pieces import AudioUnit, VideoUnit
from .select import Piece, merge_pieces

CALL2_VERSION = 6
CALL2_FILE = "call2.json"
MIN_COVER = 0.8      # a cover picture shorter than this is not worth a cut
MODES = ("synced", "cutaway")

Span = Tuple[float, float]

_COMMON = """You are a senior video editor cutting an ad down to shorter versions.
You cannot see the video. Another editor watched it and logged every piece of it for you (below).

THE AD: {summary}
CAST (who is who; "voiceover" is a narrator who is never seen)
{cast}
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
  to, refer to, or pay off something you cut. Check the ending hardest: if you cut a moment (a joke, a claim,
  a demonstration), also cut (or do not choose) the pieces that call back to it, and end on something the kept
  story has earned.
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
You get the picture and the sound as separate pieces:
  SHOTS (V01, ...)        the pictures
  SOUND SLOTS (A01, ...)  a few spoken words, or the sound between words, with the shot it was recorded over
                          ("own picture"). A sound that runs across several shots is split into one slot per
                          shot, named with a letter added to its id (for example A05a, A05b): keep ALL of
                          its slots, in order, or none; each slot gets its own picture.
Return, for each length, a list of segments in playing order. Each segment is one sound slot AND the shot that is
SEEN while it plays. Always give both. Name the slot's own picture to leave it as it was; name a different shot to
cut away.

A cutaway edit MUST cut away. An answer in which every slot keeps its own picture is wrong: that is just the
synced edit. At least two segments in every edit (more in a longer edit) must show a different shot. Do not start
from a synced edit; build it this way:
  1. Choose the SOUNDS that tell the story (the lines that must be heard).
  2. Choose the SHOTS that must be seen (the product, the reveal, the result, the closing card).
  3. Lay those shots over the chosen sounds. Good sounds to cover: voiceover and narration (nobody is seen saying
     it), a line by someone we have already seen, music or silence. Only shots marked "CAN COVER" may be used,
     and only for a slot no longer than the seconds given there; any other choice is thrown away.

The one rule that makes or breaks a cutaway: EVERY SECOND OF A SHOT CAN BE SHOWN ONLY ONCE.
- If you show a shot over one sound, then any slot you keep whose OWN picture is that same shot can no longer
  show it. Give that slot a different picture (for example a shot from a part you cut), or do not keep it.
  If you leave it on its own picture, your cutaway is thrown away.
- This comes up whenever one long sound (narration, a voiceover, music) runs over several shots: if you borrow
  one of those shots elsewhere, give that slot of the long sound another picture.
- A shot may cover several slots; it keeps playing from where it left off, up to its "CAN COVER" seconds in total.

Also:
- A shot marked "cannot cover" shows someone talking; another sound over it looks wrong. Never cut away to it.
- A sound that belongs to what is seen, like (door slams), stays with its own picture.
- Do not cover a line when seeing the person say it is the point (a punchline delivered to camera).

SHOTS
{video}

SOUND SLOTS
{audio}
"""


def name(target: float, mode: str) -> str:
    return f"{target:g}s_{mode}"


def _sid(aid: str) -> str:
    return "S" + aid[1:]


def _needs(ids: Sequence[str], as_segments: bool) -> str:
    ids = [_sid(n) if as_segments and n.startswith("A") else n for n in ids]
    return f" | needs {' '.join(ids)}" if ids else ""


def _cast(shot: Dict[str, Any]) -> str:
    return f" | cast: {' '.join(shot['cast'])}" if shot.get("cast") else ""


def _said(sound: Dict[str, Any]) -> str:
    return f"{sound['speaker']}: {sound['heard']}" if sound.get("speaker") else sound["heard"]


def synced_prompt(audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any],
                  targets: Sequence[float]) -> str:
    dv = {u["id"]: u for u in described["video_units"]}
    da = {u["id"]: u for u in described["audio_units"]}
    shots = "\n".join(f"{v.id} | {v.dur:.2f}s{_cast(dv[v.id])} | shows: {dv[v.id]['shows']}"
                      f"{_needs(dv[v.id]['needs'], True)}" for v in video)
    segs = "\n".join(f"{_sid(a.id)} | {a.dur:.2f}s | over {' '.join(a.over)} | heard: {_said(da[a.id])}"
                     f"{_needs(da[a.id]['needs'], True)}" for a in audio)
    return _fill(SYNCED_PROMPT, described, video, targets, shots=shots, segments=segs)


def cover_room(v: VideoUnit, talk: Sequence[Span]) -> float:
    """Seconds of this shot usable as a cover in one go: its longest stretch without a talking face
    (0 if that is too short to be worth a cut)."""
    longest = max((b - a for a, b in quiet_spans(v, talk)), default=0.0)
    return longest if longest >= MIN_COVER else 0.0


def cutaway_prompt(audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any],
                   targets: Sequence[float], talk: Sequence[Span] = ()) -> str:
    dv = {u["id"]: u for u in described["video_units"]}
    da = {u["id"]: u for u in described["audio_units"]}

    def can_cover(v: VideoUnit) -> str:
        room = cover_room(v, talk)
        return f"CAN COVER up to {room:.1f}s" if room else "cannot cover (a face is talking)"

    vid = "\n".join(f"{v.id} | {v.dur:.2f}s | {can_cover(v)}{_cast(dv[v.id])} | shows: {dv[v.id]['shows']}"
                    f"{_needs(dv[v.id]['needs'], False)}" for v in video)
    lines = []
    for s in make_slots(audio, video):
        heard = _said(da[s["audio"]]) if s["part"] == 0 else "(the same sound continues)"
        needs = _needs(da[s["audio"]]["needs"], False) if s["part"] == 0 else ""
        lines.append(f"{s['id']} | {s['end'] - s['start']:.2f}s | own picture {s['shot']} | heard: {heard}{needs}")
    return _fill(CUTAWAY_PROMPT, described, video, targets, video=vid, audio="\n".join(lines))


def make_slots(audio: List[AudioUnit], video: List[VideoUnit], min_part: float = 0.25) -> List[Dict[str, Any]]:
    """Every audio unit, split at the shot cuts it runs across: one slot per shot, so each part of a long
    sound can be given its own picture. Slivers shorter than ``min_part`` stay with their neighbour.
    A unit over a single shot is one slot with the unit's own id; otherwise the id plus a letter (A05a, A05b)."""
    out: List[Dict[str, Any]] = []
    for a in audio:
        parts = [[max(a.start, v.start), min(a.end, v.end), v.id] for v in video
                 if min(a.end, v.end) - max(a.start, v.start) > 1e-6]
        while len(parts) > 1:
            k = min(range(len(parts)), key=lambda j: parts[j][1] - parts[j][0])
            if parts[k][1] - parts[k][0] >= min_part:
                break
            j = k - 1 if k > 0 else k + 1
            lo, hi = min(j, k), max(j, k)
            parts[lo:hi + 1] = [[parts[lo][0], parts[hi][1], parts[j][2]]]
        if not parts:
            parts = [[a.start, a.end, a.over[0] if a.over else video[0].id]]
        for i, (s, e, shot) in enumerate(parts):
            out.append({"id": a.id if len(parts) == 1 else f"{a.id}{chr(97 + i)}", "audio": a.id,
                        "start": s, "end": e, "shot": shot, "part": i})
    return out


def _fill(template: str, described: Dict[str, Any], all_shots: List[VideoUnit], targets: Sequence[float], **tables) -> str:
    music = {"bed": "yes", "only": "yes"}.get(described.get("music"), "no")
    out = template
    cast = "\n".join(f"{cast_name(c)}: {c['description']}" for c in described.get("cast") or []) or "(nobody)"
    for key, value in {"summary": described.get("summary", ""), "music": music, "cast": cast,
                       "source": f"{all_shots[-1].end:.1f}", "targets": ", ".join(f"{t:g}s" for t in targets),
                       **tables}.items():
        out = out.replace("{" + key + "}", str(value)).replace("{" + key + ":.1f}", str(value))
    return out


# =========================================================================== the Gemini call
def request(audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any], targets: Sequence[float],
            mode: str, api_key: str, model: str = "gemini-3.8-flash", api_version: str = "v1beta",
            log: Callable[[str], None] = print, talk: Sequence[Span] = ()) -> Dict[str, Dict[str, Any]]:
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

    text = (synced_prompt(audio, video, described, targets) if mode == "synced"
            else cutaway_prompt(audio, video, described, targets, talk))
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


def _minus(spans: Sequence[Span], taken: Sequence[Span]) -> List[Span]:
    """``spans`` with every stretch in ``taken`` removed."""
    out = list(spans)
    for x, y in taken:
        out = [part for a, b in out for part in ((a, min(b, x)), (max(a, y), b)) if part[1] - part[0] > 1e-6]
    return out


def build_cut(audio: List[AudioUnit], video: List[VideoUnit], described: Dict[str, Any], edit: Dict[str, Any],
              talk: Sequence[Span] = ()) -> Dict[str, Any]:
    """``{"pieces": [...], "stats": {...}}`` for one stored edit, in the shape the renderer takes."""
    A, V = {a.id: a for a in audio}, {v.id: v for v in video}
    mode, target = edit["mode"], float(edit["target_s"])
    notes: List[str] = []
    slots = make_slots(audio, video)
    S = {s["id"]: s for s in slots}
    of_unit: Dict[str, List[str]] = {}
    for s in slots:
        of_unit.setdefault(s["audio"], []).append(s["id"])

    asked: Dict[str, str] = {}                    # slot id -> picture Gemini asked for ("" = its own)
    order: List[str] = []                         # audio units, in playing order
    for s in edit.get("segments") or []:
        sid, vid = s.get("audio", ""), s.get("video", "") if mode == "cutaway" else ""
        ids = [sid] if sid in S else of_unit.get(sid)
        if not ids:
            notes.append(f"unknown piece {sid or '(empty)'!r} ignored")
            continue
        if vid and vid not in V:
            notes.append(f"{sid}: unknown picture {vid!r}, its own picture is used")
            vid = ""
        for i in ids:
            if i in asked:
                notes.append(f"{i} listed twice: played once")
                continue
            asked[i] = "" if vid == S[i]["shot"] else vid
            if S[i]["audio"] not in order:
                order.append(S[i]["audio"])
    segs: List[List[str]] = []                    # [slot id, cover shot or ""]; a sound is never split
    for unit in order:
        missing = [i for i in of_unit[unit] if i not in asked]
        if missing:
            notes.append(f"{unit} is one sound: its other part(s) {' '.join(missing)} are kept too")
        segs.extend([i, asked.get(i, "")] for i in of_unit[unit])

    quiet = {v.id: quiet_spans(v, talk) for v in video}
    dv = {u["id"]: u for u in described["video_units"]}
    da = {u["id"]: u for u in described["audio_units"]}

    def allocate() -> List[Optional[Tuple[float, float, float]]]:
        """Per segment with a cover: (picture start, cover length, seconds of its own picture first)."""
        cursor = {v.id: v.start for v in video}
        own = [(S[i]["start"], S[i]["end"]) for i, vid in segs if not vid]      # footage shown with its own sound
        free = {v.id: _minus(quiet[v.id], own) for v in video}
        out: List[Optional[Tuple[float, float, float]]] = []
        for i, vid in segs:
            if not vid:
                out.append(None)
                continue
            need, v = S[i]["end"] - S[i]["start"], V[vid]
            win = None
            for least in (need, min(need, MIN_COVER)):   # first a quiet stretch that fits the whole sound
                for a, b in free[vid]:
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
                limit = min([v.end] + [a for a, _ in own if a >= start + length - 1e-6])
                length = min(need, limit - start)
            cursor[vid] = start + length
            out.append((start, length, max(0.0, need - length)))
        return out

    while True:                                   # undo covers that cannot work, until nothing changes
        plan = allocate()
        bad = next((k for k, (i, vid) in enumerate(segs) if vid and plan[k] is None), None)
        if bad is None:
            break
        i, vid = segs[bad]
        shown_as_own = sorted({j for j, c in segs if not c and S[j]["shot"] == vid})
        why = (f"it is already shown under {' '.join(shown_as_own)}" if shown_as_own
               else "it has no stretch without a talking face left")
        notes.append(f"{vid} cannot cover {i}: {why}. {i} keeps its own picture")
        segs[bad][1] = ""

    if mode == "cutaway" and segs and not any(vid for _, vid in segs):
        notes.append("NO CUTAWAYS: every sound plays over its own picture, so this is a synced edit")
    pieces: List[Piece] = []
    for (i, vid), win in zip(segs, plan):
        s = S[i]
        dur = s["end"] - s["start"]
        if not vid:
            pieces.append(Piece(s["start"], s["start"], dur, i, da[s["audio"]]["heard"][:60]))
            continue
        start, length, own = win
        if own > 1e-3:
            pieces.append(Piece(s["start"], s["start"], own, i, "its own picture first"))
        pieces.append(Piece(start, s["start"] + own, length, f"{i}<-{vid}", f"over: {dv[vid]['shows']}"[:80]))
    pieces = merge_pieces(pieces, bridge_gap=1e-3)

    kept = set(order)
    shown = {vid or S[i]["shot"] for i, vid in segs}
    unmet = [f"{uid} needs {n}" for uid in [*sorted(kept), *sorted(shown)]
             for n in (da.get(uid) or dv.get(uid))["needs"] if n not in kept and n not in shown]
    if unmet:
        notes.append("kept although what they need was cut: " + "; ".join(unmet))

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
            "stats": {"target": target, "mode": mode, "total": total, "units": [i for i, _ in segs],
                      "covers": {i: vid for i, vid in segs if vid}, "notes": notes, "ok": ok,
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
