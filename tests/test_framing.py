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

    def test_a_large_object_is_followed_when_nobody_is_there_whatever_its_label(self):
        bars = {"cname": "cell phone", "track_id": 1, "bbox": [186, 24, 1062, 690], "conf": 0.6}   # a product shot
        c = extract_frame_candidates(0, 0, {0: [bars]}, TEXT, {}, W, H)
        self.assertEqual([(x["type"], x["tag"]) for x in c], [("object", "object_1")])
        self.assertAlmostEqual(c[0]["cx"], 624.0)                       # the product, not the text in the corner

    def test_a_small_or_fleeting_object_is_not_a_subject(self):
        small = {"cname": "cell phone", "track_id": 5, "bbox": [648, 536, 756, 676], "conf": 0.5}
        self.assertEqual(kinds([small]), ["ocr"])
        big = {"cname": "cup", "track_id": 6, "bbox": [100, 100, 700, 600], "conf": 0.5}
        c = extract_frame_candidates(0, 0, {0: [big]}, TEXT, {}, W, H, presence={6: 0.1})
        self.assertEqual([x["type"] for x in c], ["ocr"])

    def test_a_person_comes_before_any_object(self):
        big = {"cname": "cup", "track_id": 6, "bbox": [100, 100, 700, 600], "conf": 0.9}
        self.assertEqual(kinds([big, person()]), ["person"])

    def test_movement_is_only_followed_when_there_is_nothing_else(self):
        moving = {0: {"motion_cx": 900.0, "instant_density": 0.4, "motion_weight": 0.5}}
        with_person = extract_frame_candidates(0, 0, {0: [person()]}, {}, moving, W, H)
        self.assertEqual([x["type"] for x in with_person], ["person"])
        alone = extract_frame_candidates(0, 0, {}, {}, moving, W, H)
        self.assertEqual([(x["type"], x["cx"]) for x in alone], [("motion", 900.0)])

    def test_a_shot_that_moves_slowly_falls_back_to_where_it_moves_then_to_the_middle(self):
        c = extract_frame_candidates(0, 0, {}, {}, {}, W, H, shot_motion=820.0)
        self.assertEqual([(x["tag"], x["cx"]) for x in c], [("shot_motion", 820.0)])
        c = extract_frame_candidates(0, 0, {}, {}, {}, W, H)
        self.assertEqual([(x["tag"], x["cx"]) for x in c], [("frame_center", W / 2)])

    def test_a_speaker_of_similar_size_wins_over_a_silent_person(self):
        talker = {**person(speaking=True, x=900, size=200), "track_id": 2}
        silent = {**person(x=300, size=220), "track_id": 1}
        c = extract_frame_candidates(0, 0, {0: [silent, talker]}, {}, {}, W, H)
        self.assertEqual(max(c, key=lambda x: x["score"])["tag"], "speaker_2")

    def test_an_end_card_is_framed_on_all_its_text_together(self):
        logo = {"cx": 560.0, "box_w": 160.0, "area": 30.0}
        offer = {"cx": 720.0, "box_w": 160.0, "area": 20.0}
        c = extract_frame_candidates(0, 0, {}, {0: [logo, offer]}, {}, W, H)
        self.assertEqual([(x["tag"], x["cx"]) for x in c], [("ocr_text", 640.0)])    # the middle of 480..800

    def test_text_wider_than_a_vertical_crop_is_framed_on_its_largest_box(self):
        left = {"cx": 200.0, "box_w": 200.0, "area": 10.0}
        right = {"cx": 1000.0, "box_w": 300.0, "area": 40.0}
        c = extract_frame_candidates(0, 0, {}, {0: [left, right]}, {}, W, H)
        self.assertEqual([x["cx"] for x in c], [1000.0])

    def test_text_comes_before_movement_on_an_end_card(self):
        moving = {0: {"motion_cx": 100.0, "instant_density": 0.5, "motion_weight": 0.5}}
        c = extract_frame_candidates(0, 0, {}, TEXT, moving, W, H)
        self.assertEqual([x["type"] for x in c], ["ocr"])

    def test_with_nothing_at_all_the_camera_stays_in_the_middle(self):
        c = extract_frame_candidates(5, 0, {}, TEXT, {}, W, H)           # frame 5: no people, no text
        self.assertEqual([(x["type"], x["cx"]) for x in c], [("fallback", W / 2)])


if __name__ == "__main__":
    unittest.main()
