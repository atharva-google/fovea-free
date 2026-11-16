"""Stricter speaker labels and the prominence of silent people (fovea/framing/subjects.py).

    python -m unittest discover -s tests -v
"""

import unittest

from fovea.framing.subjects import person_score, speaking_frames


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
        self.assertLessEqual(person_score(5.0, -1.0, 9.0), 0.9)      # never near a speaker's 2.0
        self.assertGreaterEqual(person_score(0.0, 1.0, 0.0), 0.5)    # never below text / motion fallbacks


if __name__ == "__main__":
    unittest.main()
