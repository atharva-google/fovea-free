"""How the camera moves inside one shot (fovea/framing/moves.py). Numbers are for a 9:16 crop of 1080p.

    python -m unittest discover -s tests -v
"""

import unittest

from fovea.framing.moves import plan_shot

CROP = 608                      # 9:16 crop width of a 1920x1080 source
LO, HI = CROP / 2, 1920 - CROP / 2


def plan(raw, tags=None, **kw):
    return plan_shot(raw, tags or ["person_1"] * len(raw), LO, HI, CROP, **kw)


def travel(cx):
    return sum(abs(b - a) for a, b in zip(cx, cx[1:]))


class CameraMovesTest(unittest.TestCase):
    def test_a_still_subject_with_detector_jitter_gives_a_still_camera(self):
        raw = [900 + (3 if i % 2 else -3) for i in range(100)]
        cx, modes = plan(raw)
        self.assertEqual(travel(cx), 0)
        self.assertEqual(set(modes[1:]), {"camera_hold"})

    def test_a_small_drift_does_not_nudge_the_camera(self):
        cx, _ = plan([900] * 40 + [960] * 60)            # 60 px: the subject is still well inside the crop
        self.assertEqual(travel(cx), 0)

    def test_a_flicker_to_another_subject_is_ignored(self):
        raw = [700] * 40 + [1300] * 6 + [700] * 54
        tags = ["person_1"] * 40 + ["speaker_2"] * 6 + ["person_1"] * 54
        cx, modes = plan(raw, tags)
        self.assertEqual(travel(cx), 0)
        self.assertNotIn("instant_cut", modes)

    def test_a_short_shot_gets_one_framing_even_if_the_subject_flips(self):
        raw = [400, 400, 1400, 1400, 400, 1400] * 3 + [400] * 5      # 23 frames, text <-> motion
        tags = ["ocr_text" if x == 400 else "action_motion" for x in raw]
        cx, modes = plan(raw, tags)
        self.assertEqual(travel(cx), 0)
        self.assertEqual(cx[0], 400)                                   # where it was followed longest
        self.assertEqual(set(modes[1:]), {"camera_hold"})

    def test_a_real_move_becomes_one_pan_that_arrives(self):
        cx, modes = plan([700] * 50 + [1100] * 70)
        self.assertEqual((cx[0], cx[49]), (700, 700))
        self.assertAlmostEqual(cx[-1], 1100, places=3)
        self.assertAlmostEqual(travel(cx), 400, places=3)              # straight there, no overshoot or return
        pans = sum(1 for a, b in zip(modes, modes[1:]) if b == "smooth_pan" and a != "smooth_pan")
        self.assertEqual(pans, 1)

    def test_a_long_move_is_slower_not_a_whip(self):
        near, _ = plan([700] * 50 + [900] * 100)
        far, _ = plan([400] * 50 + [1500] * 100)
        frames = lambda cx: sum(1 for a, b in zip(cx, cx[1:]) if a != b)   # noqa: E731
        self.assertGreater(frames(far), 2 * frames(near))
        self.assertLess(max(abs(b - a) for a, b in zip(far, far[1:])), 60)  # px per frame, at 1080p

    def test_a_lasting_switch_to_a_far_speaker_is_a_cut(self):
        raw = [500] * 50 + [1400] * 50
        tags = ["speaker_1"] * 50 + ["speaker_2"] * 50
        cx, modes = plan(raw, tags)
        self.assertEqual(modes[50], "instant_cut")
        self.assertEqual((cx[49], cx[50], cx[-1]), (500, 1400, 1400))

    def test_the_shot_opens_on_where_the_subject_settles_and_stays_inside_the_frame(self):
        cx, _ = plan([1500, 1500] + [800] * 98)          # two stray frames at the start
        self.assertEqual(cx[0], 800)
        cx, _ = plan([50] * 60 + [1900] * 60)
        self.assertEqual((min(cx), max(cx)), (LO, HI))

    def test_a_single_frame_spike_is_removed(self):
        raw = [800] * 50
        raw[20] = 1700
        self.assertEqual(travel(plan(raw)[0]), 0)

    def test_empty_and_one_frame_shots(self):
        self.assertEqual(plan([]), ([], []))
        self.assertEqual(plan([900]), ([900], ["shot_start_cut"]))


if __name__ == "__main__":
    unittest.main()
