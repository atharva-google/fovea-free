"""What the camera prefers to look at (fovea/framing/focal.py: extract_frame_candidates).

    python -m unittest discover -s tests -v
"""

import sys
import types
import unittest

for _name in ("numpy", "pandas"):       # only the scoring function is tested; it needs neither
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)

from fovea.framing.focal import extract_frame_candidates  # noqa: E402

W, H = 1280, 720
TEXT = {0: [{"cx": 1100.0, "box_w": 200.0, "area": 50.0, "confidence": 0.9}]}


def person(speaking=False, x=100, size=60):
    return {"cname": "person", "track_id": 1, "bbox": [x, 100, x + size, 100 + size],
            "face_bbox": [x, 100, x + size, 100 + size], "speaking": speaking, "asd_score": 1.0 if speaking else None}


def kinds(frame_people, shot_rel_idx=0):
    return [c["type"] for c in extract_frame_candidates(0, shot_rel_idx, {0: frame_people}, TEXT, {}, W, H)]


class WhatTheCameraFollowsTest(unittest.TestCase):
    def test_text_is_not_a_candidate_when_a_person_is_in_the_frame(self):
        self.assertEqual(kinds([person()]), ["person"])                  # even a small, silent face, at shot start
        self.assertEqual(kinds([person(speaking=True)]), ["person"])

    def test_text_is_followed_when_nobody_is_in_the_frame(self):
        self.assertEqual(kinds([]), ["ocr"])
        self.assertEqual(kinds([], shot_rel_idx=40), ["ocr"])

    def test_among_silent_people_the_central_one_who_stays_wins(self):
        centre = {**person(x=600, size=80), "track_id": 1}
        edge = {**person(x=1180, size=80), "track_id": 2}
        c = extract_frame_candidates(0, 0, {0: [edge, centre]}, {}, {}, W, H, presence={1: 1.0, 2: 0.2})
        best = max(c, key=lambda x: x["score"])
        self.assertEqual(best["tag"], "person_1")

    def test_with_nothing_at_all_the_camera_stays_in_the_middle(self):
        c = extract_frame_candidates(5, 0, {}, TEXT, {}, W, H)           # frame 5: no people, no text
        self.assertEqual([(x["type"], x["cx"]) for x in c], [("fallback", W / 2)])


if __name__ == "__main__":
    unittest.main()
