"""Pass 1 + 2: find the focal X coordinate per frame with a Viterbi path search.

Each frame produces *candidates* (speakers, persons, OCR text, motion). A
dynamic-programming search picks the sequence of candidates that balances
"interesting" (high score) against "stable" (small jumps, few subject switches).
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .subjects import fill_track_gaps, person_score, shot_motion_cx, speaker_score, watermark_indices

OcrByFrame = Dict[int, List[Dict[str, float]]]


def _vertices(val):
    if isinstance(val, (list, tuple)):
        return val
    if isinstance(val, str):
        val_clean = val.strip()
        try:
            return ast.literal_eval(val_clean)
        except (ValueError, SyntaxError):
            matches = re.findall(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", val_clean)
            if matches:
                return [(float(x), float(y)) for x, y in matches]
    return []


def split_logos(ocr_df: pd.DataFrame, n_shots: Optional[int] = None):
    """Pre-check, before any crop centre is computed: text that sits in the same place in several shots
    is a logo / watermark. Returns ``(text to frame, logos)``; only the first goes on to the framing."""
    if len(ocr_df) == 0:
        return ocr_df, ocr_df
    where = []
    for k, (_, row) in enumerate(ocr_df.iterrows()):
        v = _vertices(row["text_box_vertices"])
        if not v or len(v) < 2:
            where.append((f"unreadable-{k}", -1.0 - k, -1.0 - k))     # never matches another box
            continue
        xs, ys = [p[0] for p in v], [p[1] for p in v]
        where.append((row["shot_id"], (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0))
    logos = watermark_indices(where, n_shots)
    is_logo = [k in logos for k in range(len(ocr_df))]
    keep = [not x for x in is_logo]
    return ocr_df[keep].reset_index(drop=True), ocr_df[is_logo].reset_index(drop=True)


def parse_ocr_entries(ocr_df: pd.DataFrame, fps: float, video_w: int) -> OcrByFrame:
    """Expand per-shot OCR rows into a {frame_idx: [boxes]} lookup."""
    ocr_by_frame: OcrByFrame = {}
    for _, row in ocr_df.iterrows():
        f_start = int(round(float(row["start_s"]) * fps))
        f_end = int(round(float(row["end_s"]) * fps))

        vertices = _vertices(row["text_box_vertices"])
        if not vertices or len(vertices) < 2:
            continue

        xs = [v[0] * video_w for v in vertices]
        ys = [v[1] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        box_w = max_x - min_x
        box_h = max(ys) - min(ys)

        entry = {
            "cx": (min_x + max_x) / 2.0,
            "box_w": box_w,
            "area": box_w * box_h,
            "confidence": float(row.get("confidence", 1.0)),
        }
        for f in range(f_start, f_end + 1):
            ocr_by_frame.setdefault(f, []).append(entry)
    return ocr_by_frame


def extract_frame_candidates(
    f_idx: int,
    shot_rel_idx: int,
    frames_data: Dict[int, List[Dict[str, Any]]],
    ocr_by_frame: OcrByFrame,
    motion_results: Dict[int, Dict[str, float]],
    video_w: int,
    video_h: int,
    presence: Optional[Dict[Any, float]] = None,
    shot_motion: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """What the camera could look at on this frame. One kind at a time, in this order:

        faces  >  a person with no visible face  >  a large object  >  on-screen text  >  movement
        >  the middle of the frame

    so a lower kind is only ever followed when nothing of a higher kind is there.
    ``presence``: share of the shot each track id is on screen for (missing = the whole shot).
    ``shot_motion``: where the movement of the whole shot is (``subjects.shot_motion_cx``), if any.
    """
    candidates = []
    frame_area = float(video_w * video_h)
    presence = presence or {}
    dets = frames_data.get(f_idx, [])

    # 1. Faces: ranked by the size of the face, how central and in focus it is, and how long it is in
    #    the shot. Speaking is a boost on top, scaled by the face's size next to the largest face.
    persons = [d for d in dets if d.get("cname") == "person"]
    sharpest = max((d.get("sharp") or 0.0 for d in persons), default=0.0)

    def face_area(d):
        b = d.get("face_bbox")
        return float((b[2] - b[0]) * (b[3] - b[1])) if b is not None else 0.0

    largest_face = max((face_area(d) for d in persons), default=0.0)
    for det in persons:
        tid = det["track_id"]
        box = det.get("face_bbox") or det["bbox"]
        cx = (box[0] + box[2]) / 2.0
        offset = abs(cx / float(video_w) - 0.5) / 0.5
        rel_sharp = det["sharp"] / sharpest if det.get("sharp") is not None and sharpest > 0 else 1.0
        if det.get("face_bbox") is not None:
            score = person_score(face_area(det) / frame_area, offset, presence.get(tid, 1.0), rel_sharp, full=0.10)
        else:   # older analysis: a person stored without a face box
            area = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
            score = person_score(area / frame_area, offset, presence.get(tid, 1.0))
        if det.get("speaking") is True:
            rel = face_area(det) / largest_face if largest_face > 0 else 1.0
            score = speaker_score(score, rel, det.get("asd_score") or 0.0)
            tag = f"speaker_{tid}"
        else:
            tag = f"person_{tid}"
        candidates.append({"cx": cx, "score": score, "tag": tag, "type": "person"})
    if candidates:
        return candidates

    # 2. A person with no visible face (seen from behind, far away): only when no face is there.
    for det in dets:
        if det.get("cname") != "body":
            continue
        x1, y1, x2, y2 = det["bbox"]
        offset = abs((x1 + x2) / 2.0 / float(video_w) - 0.5) / 0.5
        score = 0.9 * person_score((x2 - x1) * (y2 - y1) / frame_area, offset, presence.get(det["track_id"], 1.0))
        candidates.append({"cx": (x1 + x2) / 2.0, "score": score, "tag": f"body_{det['track_id']}", "type": "person"})
    if candidates:
        return candidates

    # 3. A large object that is there for a real part of the shot (a product). The detector's label is
    #    ignored: it calls a chocolate bar a "cell phone", but the box is right.
    for det in dets:
        if det.get("cname") in ("person", "body"):
            continue
        x1, y1, x2, y2 = det["bbox"]
        ratio = (x2 - x1) * (y2 - y1) / frame_area
        if ratio >= 0.05 and presence.get(det["track_id"], 1.0) >= 0.3:
            candidates.append({"cx": (x1 + x2) / 2.0, "score": 0.40 + 0.08 * min(1.0, ratio / 0.25),
                               "tag": f"object_{det['track_id']}", "type": "object"})
    if candidates:
        return candidates

    # 4. On-screen text (logos that sit in the same place in several shots were removed before framing).
    #    High for the first ~0.5s of a shot, then it decays once read.
    if f_idx in ocr_by_frame:
        read_decay = 1.0 if shot_rel_idx < 12 else max(0.15, 1.0 - (shot_rel_idx - 12) * 0.08)
        return [{"cx": box["cx"], "score": 0.85 * read_decay, "tag": "ocr_text", "type": "ocr"}
                for box in ocr_by_frame[f_idx]]

    # 5. Movement: strong movement on this frame, else where the whole shot moves.
    motion = motion_results.get(f_idx, {})
    density = motion.get("instant_density", 0.0)
    weight = motion.get("motion_weight", 0.0)
    mcx = motion.get("motion_cx")
    if mcx is not None and (density >= 0.10 or weight >= 0.15) and 0.05 * video_w <= mcx <= 0.95 * video_w:
        return [{"cx": mcx, "score": 1.50 * (density + 0.5 * weight), "tag": "action_motion", "type": "motion"}]
    if shot_motion is not None and 0.05 * video_w <= shot_motion <= 0.95 * video_w:
        return [{"cx": shot_motion, "score": 0.30, "tag": "shot_motion", "type": "motion"}]

    # 6. Nothing to go on: the middle of the frame.
    return [{"cx": video_w / 2.0, "score": 0.20, "tag": "frame_center", "type": "fallback"}]


def optimize_shot_focal_path(
    shot_frames: List[int],
    frames_data: Dict[int, List[Dict[str, Any]]],
    ocr_by_frame: OcrByFrame,
    motion_results: Dict[int, Dict[str, float]],
    video_w: int,
    video_h: int,
    max_jump_cost: float = 4.0,
    switch_penalty: float = 1.2,
) -> List[Dict[str, Any]]:
    T = len(shot_frames)
    if T == 0:
        return []

    frames_data = fill_track_gaps(frames_data, shot_frames)   # someone lost for a few frames is still there
    seen: Dict[Any, int] = {}                 # how long each person / object is on screen in this shot
    for f in shot_frames:
        for det in frames_data.get(f, []):
            seen[det["track_id"]] = seen.get(det["track_id"], 0) + 1
    presence = {tid: n / float(T) for tid, n in seen.items()}
    shot_motion = shot_motion_cx(motion_results, shot_frames)

    frame_candidates = [
        extract_frame_candidates(f, t, frames_data, ocr_by_frame, motion_results, video_w, video_h, presence,
                                 shot_motion)
        for t, f in enumerate(shot_frames)
    ]

    dp_cost = [[-np.log(max(1e-4, c["score"])) for c in frame_candidates[0]]]
    backpointer = [[-1] * len(frame_candidates[0])]

    for t in range(1, T):
        prev_cands, curr_cands = frame_candidates[t - 1], frame_candidates[t]
        t_costs, t_bp = [], []
        for curr_c in curr_cands:
            obs_cost = -np.log(max(1e-4, curr_c["score"]))
            best_prev_idx, best_total = -1, float("inf")
            for prev_idx, prev_c in enumerate(prev_cands):
                # Linear transition cost capped at max_jump_cost
                norm_dx = abs(curr_c["cx"] - prev_c["cx"]) / float(video_w)
                trans_cost = min(max_jump_cost, 6.0 * norm_dx)
                if curr_c["tag"] != prev_c["tag"]:
                    trans_cost += switch_penalty
                total = dp_cost[t - 1][prev_idx] + obs_cost + trans_cost
                if total < best_total:
                    best_total, best_prev_idx = total, prev_idx
            t_costs.append(best_total)
            t_bp.append(best_prev_idx)
        dp_cost.append(t_costs)
        backpointer.append(t_bp)

    # Backtrack
    path = [int(np.argmin(dp_cost[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(backpointer[t][path[-1]])
    path.reverse()

    return [
        {
            "frame_idx": shot_frames[t],
            "target_cx": round(frame_candidates[t][ci]["cx"], 2),
            "tag": frame_candidates[t][ci]["tag"],
            "type": frame_candidates[t][ci]["type"],
        }
        for t, ci in enumerate(path)
    ]


def find_focal_points_viterbi(
    shots_df: pd.DataFrame,
    shot_tracks: Dict[int, Dict[int, List[Dict[str, Any]]]],
    ocr_df: pd.DataFrame,
    motion_results: Dict[int, Dict[str, float]],
    video_width: int,
    video_height: int,
    fps: float = 25.0,
    max_jump_cost: float = 4.0,
    switch_penalty: float = 1.2,
) -> pd.DataFrame:
    ocr_by_frame = parse_ocr_entries(ocr_df, fps, video_width)
    records = []
    for _, shot in shots_df.iterrows():
        s_id = int(shot["shot_id"])
        shot_frames = list(range(int(shot["start_frame"]), int(shot["end_frame"])))
        for entry in optimize_shot_focal_path(
            shot_frames, shot_tracks.get(s_id, {}), ocr_by_frame, motion_results,
            video_width, video_height, max_jump_cost, switch_penalty,
        ):
            entry["shot_id"] = s_id
            records.append(entry)
    return pd.DataFrame(records)
