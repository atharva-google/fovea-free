"""Pass 3: Kinematic camera director.

Turns raw per-frame focal points into camera moves: stationary holds, smooth
cosine-eased pans, or instant cuts. Works for any crop width (any aspect ratio).
The rules for one shot are in ``moves.py`` (plain Python, tested in tests/test_moves.py).
"""

from __future__ import annotations

import pandas as pd

from .geometry import CropGeometry
from .moves import plan_shot


def plan_camera(
    focal_df: pd.DataFrame,
    geometry: CropGeometry,
    pan_window_frames: int = 12,
    cut_threshold_px: float = 450.0,
    still_deadband_px: float = 45.0,
    spike_threshold_px: float = 200.0,
) -> pd.DataFrame:
    min_cx, max_cx = geometry.min_cx, geometry.max_cx
    records = []

    for shot_id, group in focal_df.groupby("shot_id", sort=False):
        group = group.sort_values("frame_idx")
        raw = [float(x) for x in group["target_cx"].tolist()]
        tags = group["tag"].tolist()
        frames = group["frame_idx"].tolist()
        cx, modes = plan_shot(raw, tags, min_cx, max_cx, geometry.crop_w, pan_window_frames, cut_threshold_px,
                              still_deadband_px, spike_threshold_px)
        for idx in range(len(raw)):
            records.append({
                "shot_id": shot_id,
                "frame_idx": int(frames[idx]),
                "target_cx": round(raw[idx], 2),
                "smoothed_cx": round(float(cx[idx]), 2),
                "camera_mode": modes[idx],
                "tag": tags[idx],
            })
    return pd.DataFrame(records)
