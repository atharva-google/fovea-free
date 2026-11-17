"""Cut one video into jigsaw pieces, on two tracks that each cover the whole video. Code only.

    audio units  A01, A02, ...  a few spoken words, or the sound between them ("gap").
                                Split at every audible pause and at punctuation. A phrase that runs
                                across a shot cut stays ONE unit (a word cannot be split).
    video units  V01, V02, ...  one per shot: a shot is a visual unit. Each carries how much of it
                                shows someone visibly talking.
    AV segments  S01, S02, ...  for the synced edit: one audio unit with the picture under it.

Gemini only ever names these ids; every time used later comes from here.
No third-party imports (tested in tests/test_pieces.py).
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PIECES_VERSION = 2  # bump when the cutting rules change in a way that makes stored pieces stale

_PUNCT = re.compile(r"[.?!,;:\u0964]$")  # \u0964 = the Devanagari full stop

Span = Tuple[float, float]


@dataclass
class AudioUnit:
    id: str
    start: float
    end: float
    kind: str                                   # "speech" | "gap"
    text: str = ""                              # rough transcript (Whisper); "" for a gap
    words: List[Dict[str, Any]] = field(default_factory=list)   # [{"start", "end", "word"}]
    over: List[str] = field(default_factory=list)               # video units it was recorded over

    @property
    def dur(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AudioUnit":
        return cls(str(d["id"]), float(d["start"]), float(d["end"]), str(d["kind"]), str(d.get("text") or ""),
                   list(d.get("words") or []), list(d.get("over") or []))


@dataclass
class VideoUnit:
    id: str
    start: float
    end: float
    shot: int                                   # position in the time-sorted shot list
    talking: float = 0.0                        # share of the shot in which a face is visibly talking (0-1)
    ocr: List[str] = field(default_factory=list)  # on-screen text the detector read in this shot

    @property
    def dur(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoUnit":
        return cls(str(d["id"]), float(d["start"]), float(d["end"]), int(d["shot"]),
                   float(d.get("talking") or 0.0), list(d.get("ocr") or []))


# =========================================================================== audio units
def build_audio_units(words: Sequence[Dict[str, Any]], shots: Sequence[Span], min_gap: float = 0.15,
                      max_gap: float = 2.5, pad: float = 0.06) -> List[AudioUnit]:
    """Tile ``[0, end of the last shot]`` with speech and gap units.

    ``min_gap``: a pause at least this long is its own unit (so it can be cut); shorter pauses are
    shared between the words around them. ``max_gap``: longer silences are split into chunks.
    ``pad``: a little air kept around speech when the pause next to it becomes its own unit.
    """
    if not shots:
        return []
    total = float(shots[-1][1])
    ws = sorted(({"start": max(0.0, float(w["start"])), "end": min(total, float(w["end"])), "word": str(w["word"])}
                 for w in words), key=lambda w: w["start"])
    ws = [w for w in ws if w["end"] > w["start"]]

    phrases: List[List[Dict[str, Any]]] = []
    for w in ws:
        if phrases:
            prev = phrases[-1][-1]
            if w["start"] - prev["end"] < min_gap and not _PUNCT.search(prev["word"].strip()):
                phrases[-1].append(w)
                continue
        phrases.append([w])

    # (start, end, words) per phrase, forced into time order without overlaps
    spans: List[List[Any]] = []
    for ph in phrases:
        s, e = ph[0]["start"], max(w["end"] for w in ph)
        if spans:
            s = max(s, spans[-1][1])
            if e - s < 0.02:            # nothing left of it after its neighbour: same phrase
                spans[-1][2].extend(ph)
                continue
        spans.append([s, e, list(ph)])

    raw: List[Tuple[float, float, str, List[Dict[str, Any]]]] = []
    cursor = 0.0                         # where the next unit starts
    for i, (s, e, ph) in enumerate(spans):
        first = i == 0
        g = s - (0.0 if first else spans[i - 1][1])
        if g >= min_gap:                 # the pause before this phrase is its own unit
            start = s - min(pad, g / 4)
            raw.extend((a, b, "gap", []) for a, b in _split_gap(cursor, start, shots, min_gap, max_gap))
        else:
            start = cursor
        nxt = spans[i + 1][0] if i + 1 < len(spans) else total
        g_after = nxt - e
        if g_after >= min_gap:
            end = e + min(pad, g_after / 4)
        else:                            # share a short pause with the next phrase / run to the end
            end = (e + nxt) / 2 if i + 1 < len(spans) else total
        raw.append((start, end, "speech", ph))
        cursor = end
    if total - cursor > 1e-6:
        raw.extend((a, b, "gap", []) for a, b in _split_gap(cursor, total, shots, min_gap, max_gap))

    return [AudioUnit(f"A{i + 1:02d}", round(a, 3), round(b, 3), kind,
                      " ".join(w["word"].strip() for w in ph), ph)
            for i, (a, b, kind, ph) in enumerate(raw)]


def _split_gap(a: float, b: float, shots: Sequence[Span], min_gap: float, max_gap: float) -> List[Span]:
    """A silence, split at the shot cuts inside it and then into chunks of at most ``max_gap``."""
    cuts = [a] + [s for s, _ in shots if a + min_gap <= s <= b - min_gap] + [b]
    out: List[Span] = []
    for x, y in zip(cuts, cuts[1:]):
        n = max(1, math.ceil((y - x) / max_gap - 1e-9))
        step = (y - x) / n
        out.extend((x + k * step, x + (k + 1) * step) for k in range(n))
    return out


# =========================================================================== video units
def build_video_units(shots: Sequence[Span], talk: Sequence[Span] = (),
                      ocr_by_shot: Optional[Dict[int, List[str]]] = None) -> List[VideoUnit]:
    """One video unit per shot. ``talk`` = time spans in which a face is visibly talking
    (``spans_from_frames``)."""
    ocr_by_shot = ocr_by_shot or {}
    units = []
    for i, (a, b) in enumerate(shots):
        a, b = float(a), float(b)
        if b <= a:
            continue
        talked = sum(max(0.0, min(b, float(y)) - max(a, float(x))) for x, y in talk)
        units.append(VideoUnit(f"V{len(units) + 1:02d}", round(a, 3), round(b, 3), i,
                               round(min(1.0, talked / (b - a)), 2), list(ocr_by_shot.get(i, []))))
    return units


def spans_from_frames(frames: Iterable[int], fps: float, max_hole: float = 0.2) -> List[Span]:
    """Frame numbers in which something is true -> time spans, bridging holes up to ``max_hole`` s."""
    out: List[List[float]] = []
    for f in sorted(set(int(x) for x in frames)):
        a, b = f / fps, (f + 1) / fps
        if out and a - out[-1][1] <= max_hole:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


# =========================================================================== linking the tracks
def link(audio: List[AudioUnit], video: List[VideoUnit], min_overlap: float = 0.1) -> None:
    """Fill ``over``: the video units each audio unit was recorded over (in time order)."""
    for a in audio:
        laps = [(min(a.end, v.end) - max(a.start, v.start), v.id) for v in video]
        a.over = [vid for lap, vid in laps if lap > min_overlap]
        if not a.over and laps:
            a.over = [max(laps)[1]]


def cut_pieces(shots: Sequence[Span], words: Sequence[Dict[str, Any]], talk: Sequence[Span] = (),
               ocr_by_shot: Optional[Dict[int, List[str]]] = None) -> Tuple[List[AudioUnit], List[VideoUnit]]:
    """Both tracks for one video, linked."""
    audio = build_audio_units(words, shots)
    video = build_video_units(shots, talk, ocr_by_shot)
    link(audio, video)
    return audio, video


def av_segments(audio: List[AudioUnit], video: List[VideoUnit], heard: Optional[Dict[str, str]] = None,
                shows: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """For the synced edit: one segment per audio unit, with the picture(s) it was recorded over.

    ``heard`` / ``shows``: Gemini's call-1 descriptions by unit id; without them the rough
    transcript is used and ``shows`` stays empty.
    """
    heard, shows = heard or {}, shows or {}
    by_id = {v.id: v for v in video}
    out = []
    for i, a in enumerate(audio):
        out.append({
            "id": f"S{i + 1:02d}", "audio": a.id, "video": list(a.over),
            "start": a.start, "end": a.end, "length": round(a.dur, 3),
            "heard": heard.get(a.id) or (a.text if a.kind == "speech" else "(no speech)"),
            "shows": [shows[v] for v in a.over if shows.get(v)],
            "shots": sorted({by_id[v].shot for v in a.over}),
        })
    return out


# =========================================================================== for people
def pieces_table(audio: List[AudioUnit], video: List[VideoUnit]) -> str:
    lines = ["AUDIO UNITS"]
    for a in audio:
        said = f'"{a.text}"' if a.kind == "speech" else "(gap)"
        lines.append(f"  {a.id}  {a.start:6.2f}-{a.end:6.2f}s {a.dur:5.2f}s  over {' '.join(a.over):<8} {said}")
    lines.append("VIDEO UNITS")
    for v in video:
        text = f"  text: {'; '.join(v.ocr)}" if v.ocr else ""
        lines.append(f"  {v.id}  {v.start:6.2f}-{v.end:6.2f}s {v.dur:5.2f}s  shot {v.shot:<3} "
                     f"visibly talking {v.talking:.0%}{text}")
    return "\n".join(lines)
