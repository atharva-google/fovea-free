"""Optional: Gemini checks our framing. Its own call, its own stored answer; call 1 is not involved.

Gemini is shown a copy of the video on which a green line marks where our software would centre
the picture, with the shot id in the corner (``steps.render.focus_preview``). Per shot it either
accepts the line or gives a fixed position to use for that shot instead.

Off unless asked for (``--gemini-framing``): sometimes our line is the better one, and sometimes
Gemini misses a shot it should have corrected. The answer is stored in prep/framing_check.json
together with a signature of the line it judged, so it is reused only for that exact line.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .describe import ask_about_video
from .pieces import VideoUnit

FRAMING_VERSION = 1
FRAMING_FILE = "framing_check.json"

PROMPT = """You are checking the framing of a video that will be cropped to a narrower format (for example vertical).

A GREEN VERTICAL LINE has been drawn on the video by our software. It marks where the software would centre the
cropped picture. The line can move within a shot, for example from one speaker to the other. The id of the
current shot (V01, V02, ...) is written in the top-left corner. Neither is part of the video itself.

For EVERY shot, judge the green line:
  fixed_focus_x  -1 if the line is on, or close to, what the viewer must see in this shot. This is the usual case.
                 A line that follows the right people as they take turns speaking is fine: do not replace it.
                 ONLY if the line is clearly on the wrong thing for most of the shot (a person in the background,
                 a logo or watermark, an empty part of the frame, the wrong one of two people) give where the
                 picture should be centred instead, for the WHOLE shot: left to right, as a percentage of the frame
                 width (0 = left edge, 50 = middle, 100 = right edge).
  why            when you give a position: what the line is on, and what it should be on. Otherwise empty.

Be conservative: a wrong correction is worse than none. If you are not sure, answer -1.
Refer to shots ONLY by their id.

SHOTS
{shots}
"""


def prompt(video: Sequence[VideoUnit]) -> str:
    return PROMPT.replace("{shots}", "\n".join(f"{v.id} | {v.start:.2f}-{v.end:.2f}s ({v.dur:.2f}s)" for v in video))


def signature(centres: Sequence[float]) -> str:
    """Identifies the green line that was judged (its position per frame, to the pixel)."""
    return hashlib.md5(",".join(str(int(round(x))) for x in centres).encode()).hexdigest()


def clean(raw: Dict[str, Any], video: Sequence[VideoUnit]) -> List[Dict[str, Any]]:
    """One entry per shot, ``fixed_focus_x`` in 0-100 or -1 (also for shots Gemini left out)."""
    got = {str(s.get("id", "")).strip(): s for s in raw.get("shots") or [] if isinstance(s, dict)}
    out = []
    for v in video:
        s = got.get(v.id, {})
        try:
            x = int(round(float(s.get("fixed_focus_x", -1))))
        except (TypeError, ValueError):
            x = -1
        x = x if 0 <= x <= 100 else -1
        out.append({"id": v.id, "fixed_focus_x": x, "why": str(s.get("why") or "") if x >= 0 else ""})
    return out


def request(preview_path: str | Path, video: Sequence[VideoUnit], api_key: str, model: str = "gemini-3.8-flash",
            api_version: str = "v1beta", log: Callable[[str], None] = print) -> Dict[str, Any]:
    raw, seconds = ask_about_video(preview_path, prompt(video), "FramingCheck", api_key, model, api_version, log,
                                   "framing check")
    return {"shots": clean(raw, video), "model": model, "seconds": seconds}


def focus_fixes(video: Sequence[VideoUnit], shots: Sequence[Dict[str, Any]]) -> List[Tuple[float, float, int]]:
    """The shots Gemini corrected: ``[(start_s, end_s, x percent)]``."""
    by_id = {s["id"]: s for s in shots}
    return [(v.start, v.end, by_id[v.id]["fixed_focus_x"]) for v in video
            if v.id in by_id and by_id[v.id].get("fixed_focus_x", -1) >= 0]


def save(folder: str | Path, fp: str, line: str, result: Dict[str, Any]) -> None:
    path = Path(folder) / FRAMING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"fingerprint": fp, "framing_version": FRAMING_VERSION, "line": line, **result},
                              indent=1, ensure_ascii=False))
    tmp.replace(path)


def load(folder: str | Path, fp: str, line: str) -> Optional[Dict[str, Any]]:
    """The stored answer, if it is for this exact file and this exact green line."""
    try:
        d = json.loads((Path(folder) / FRAMING_FILE).read_text())
    except (OSError, ValueError):
        return None
    if d.get("fingerprint") != fp or d.get("framing_version") != FRAMING_VERSION or d.get("line") != line:
        return None
    return d if isinstance(d.get("shots"), list) else None


def report(video: Sequence[VideoUnit], shots: Sequence[Dict[str, Any]]) -> str:
    by_id = {s["id"]: s for s in shots}
    fixed = [(v, by_id[v.id]) for v in video if by_id.get(v.id, {}).get("fixed_focus_x", -1) >= 0]
    lines = [f"  Gemini corrected the focal point in {len(fixed)} of {len(video)} shots"]
    lines += [f"    {v.id} {v.start:.1f}-{v.end:.1f}s -> centre at {s['fixed_focus_x']}%: {s['why']}" for v, s in fixed]
    return "\n".join(lines)
