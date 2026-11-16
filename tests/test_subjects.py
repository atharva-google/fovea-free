"""Stricter speaker labels and the prominence of silent people (fovea/framing/subjects.py).

    python -m unittest discover -s tests -v
"""

import unittest

from fovea.framing.subjects import (fill_track_gaps, person_score, shot_motion_cx, speaker_score,
                                    speaking_frames, watermark_indices)


def track(tid, scores, shot=0, start=0):
    return {"shot_id": shot, "track_id": tid, "frames": list(range(start, start + len(scores))), "scores": scores}


def frames_of(result, tid):
    return sorted(f for (_, f, t) in result if t == tid)


class SpeakingTest(unittest.TestCase):
    def test_sustained_speech_is_kept(self):
        got = speaking_frames([track(1, [-1.0] * 10 + [1.5] * 40 + [-1.0] * 10)])
        self.assertGreaterEqual(len(frames_of(got, 1)), 36)
        self.assertTrue(set(range(15, 45)) <= set(frames_of(got, 1)))

    def test_a_stray_positive_frame_is_not_speech(self):
        scores = [-1.0] * 60
        scores[20] = scores[21] = 2.0                    # two frames of noise on a silent face
        self.assertEqual(speaking_frames([track(1, scores)]), {})

    def test_a_short_dip_inside_speech_does_not_break_it(self):
        got = speaking_frames([track(1, [1.5] * 20 + [-3.0] * 4 + [1.5] * 20)])
        self.assertEqual(frames_of(got, 1), list(range(44)))

    def test_only_the_strongest_face_speaks_at_a_time(self):
        a = track(1, [2.0] * 30 + [-1.0] * 30)
        b = track(2, [0.4] * 30 + [1.8] * 30)            # weakly positive while A talks, then really talks
        got = speaking_frames([a, b])
        self.assertTrue(set(range(0, 25)) <= set(frames_of(got, 1)))
        self.assertTrue(set(range(35, 60)) <= set(frames_of(got, 2)))
        self.assertFalse(set(frames_of(got, 2)) & set(range(0, 25)))

    def test_a_tiny_background_face_is_never_the_speaker(self):
        area = {(0, f, 1): 100 * 100 for f in range(40)}
        area.update({(0, f, 2): 30 * 30 for f in range(40)})
        got = speaking_frames([track(1, [-1.0] * 40), track(2, [2.0] * 40)], area)
        self.assertEqual(got, {})
        got = speaking_frames([track(2, [2.0] * 40)], {(0, f, 2): 30 * 30 for f in range(40)})
        self.assertEqual(len(frames_of(got, 2)), 40)      # small, but the only face: it may speak

    def test_a_face_a_quarter_the_size_of_the_main_face_is_not_the_speaker(self):
        # the frame from a real ad: a large central face (238x237) and a man behind (123x122) scored as speaking
        area = {(0, f, 1): 238 * 237 for f in range(40)}
        area.update({(0, f, 3): 123 * 122 for f in range(40)})
        got = speaking_frames([track(1, [-0.5] * 40), track(3, [1.5] * 40)], area)
        self.assertEqual(got, {})

    def test_barely_positive_is_not_speech(self):
        self.assertEqual(speaking_frames([track(1, [0.05] * 40)]), {})

    def test_shots_do_not_mix(self):
        got = speaking_frames([track(1, [2.0] * 30, shot=0), track(1, [2.0] * 30, shot=1)])
        self.assertEqual({s for (s, _, _) in got}, {0, 1})


class ProminenceTest(unittest.TestCase):
    def test_large_central_and_present_beats_small_at_the_edge_and_passing(self):
        main = person_score(area_ratio=0.25, center_offset=0.1, presence=1.0)
        passer = person_score(area_ratio=0.03, center_offset=0.9, presence=0.2)
        self.assertGreater(main, passer + 0.25)

    def test_each_factor_counts_and_a_speaker_still_wins(self):
        base = person_score(0.1, 0.5, 0.5)
        self.assertGreater(person_score(0.2, 0.5, 0.5), base)
        self.assertGreater(person_score(0.1, 0.1, 0.5), base)
        self.assertGreater(person_score(0.1, 0.5, 1.0), base)
        self.assertLessEqual(person_score(5.0, -1.0, 9.0), 0.9)
        self.assertGreater(person_score(0.0, 1.0, 0.0, rel_sharp=0.0), 0.2)   # still positive

    def test_a_blurred_face_ranks_well_below_the_same_face_in_focus(self):
        sharp = person_score(0.05, 0.3, 1.0, rel_sharp=1.0, full=0.10)
        blurred = person_score(0.05, 0.3, 1.0, rel_sharp=0.2, full=0.10)
        self.assertLess(blurred, 0.7 * sharp)

    def test_the_large_central_face_beats_a_face_in_the_crowd_behind(self):
        # the frame from a real ad (1280x720): hero face 238x237 near the middle, a man behind 123x122 on the right
        frame = 1280 * 720.0
        hero = person_score(238 * 237 * 1.69 / frame, 0.2, 1.0, full=0.10)       # 1.69: the stored margin
        crowd = person_score(123 * 122 * 1.69 / frame, 0.42, 1.0, full=0.10)
        self.assertGreater(hero, crowd + 0.2)


class SpeakerBoostTest(unittest.TestCase):
    def test_in_a_conversation_between_similar_faces_the_speaker_clearly_wins(self):
        silent = person_score(0.2, 0.3, 1.0)
        self.assertGreater(speaker_score(person_score(0.2, 0.4, 1.0), rel_face=0.9, asd=0.8), 1.8 * silent)

    def test_speaking_helps_a_small_face_much_less(self):
        base = person_score(0.2, 0.3, 1.0)
        self.assertLess(speaker_score(base, rel_face=0.3), 1.15 * base)
        self.assertGreater(speaker_score(base, rel_face=1.0), 2.0 * base)


def det(tid, x, cname="person", w=100, face=True):
    d = {"track_id": tid, "cname": cname, "bbox": [x, 100, x + w, 400], "conf": 0.8}
    if face:
        d["face_bbox"] = [x + 10, 110, x + 60, 170]
    return d


class GapsTest(unittest.TestCase):
    def test_a_person_lost_between_two_sightings_is_filled_in(self):
        data = {10: [det(1, 100)], 14: [det(1, 140)]}
        out = fill_track_gaps(data, range(10, 15))
        self.assertEqual([len(out[f]) for f in range(10, 15)], [1] * 5)
        self.assertEqual(out[12][0]["bbox"][0], 120)                 # moves evenly
        self.assertEqual(out[12][0]["face_bbox"][0], 130)
        self.assertTrue(out[12][0]["filled"])
        self.assertNotIn(12, data)                                   # the input is not changed

    def test_frames_with_nobody_borrow_the_nearest_people(self):
        # the pattern from a real close-up: nobody, a person, a long dropout, the person again
        data = {f: [det(1, 0)] for f in [533, 534, 535, 536, 551, 552]}
        out = fill_track_gaps(data, range(531, 557))
        self.assertTrue(all(any(d["cname"] == "person" for d in out[f]) for f in range(531, 557)))
        self.assertTrue(out[531][0]["filled"] and out[556][0]["filled"])
        self.assertFalse(out[533][0].get("filled"))

    def test_a_long_absence_is_not_filled_and_a_filled_person_never_speaks(self):
        data = {0: [{**det(1, 100), "speaking": True, "asd_score": 1.0}], 60: [det(1, 100)]}
        out = fill_track_gaps(data, range(0, 61))
        self.assertEqual(out[30], [])
        self.assertFalse(out[10][0]["speaking"])                     # borrowed for the first 25 frames only
        self.assertEqual(out[26], [])


class WatermarkTest(unittest.TestCase):
    def test_text_in_the_same_corner_in_many_shots_is_a_logo(self):
        boxes = [(s, 0.10, 0.15) for s in range(8)] + [(3, 0.5, 0.8), (5, 0.52, 0.2)]
        self.assertEqual(watermark_indices(boxes, n_shots=20), set(range(8)))

    def test_text_seen_in_one_or_two_shots_is_kept(self):
        self.assertEqual(watermark_indices([(0, 0.5, 0.5), (1, 0.5, 0.5), (2, 0.1, 0.9)], n_shots=10), set())
        self.assertEqual(watermark_indices([], n_shots=10), set())

    def test_three_shots_are_enough_however_long_the_video(self):
        boxes = [(s, 0.1, 0.1) for s in range(3)]
        self.assertEqual(watermark_indices(boxes, n_shots=40), {0, 1, 2})
        self.assertEqual(watermark_indices(boxes[:2], n_shots=40), set())

    def test_the_same_text_twice_in_one_shot_is_not_three_shots(self):
        self.assertEqual(watermark_indices([(7, 0.1, 0.1)] * 5, n_shots=10), set())


class ShotMotionTest(unittest.TestCase):
    def test_slow_movement_adds_up_over_the_shot(self):
        motion = {f: {"motion_cx": 900.0, "instant_density": 0.07, "motion_weight": 0.0} for f in range(30)}
        self.assertAlmostEqual(shot_motion_cx(motion, range(30)), 900.0)

    def test_a_still_shot_or_a_brief_flicker_has_no_motion(self):
        still = {f: {"motion_cx": 640.0, "instant_density": 0.01, "motion_weight": 0.0} for f in range(30)}
        self.assertIsNone(shot_motion_cx(still, range(30)))
        flicker = {**still, 5: {"motion_cx": 100.0, "instant_density": 0.5, "motion_weight": 0.5}}
        self.assertIsNone(shot_motion_cx(flicker, range(30)))
        self.assertIsNone(shot_motion_cx({}, range(30)))


if __name__ == "__main__":
    unittest.main()
