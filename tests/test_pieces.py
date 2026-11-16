"""Cutting a video into audio units, video units and AV segments. No video, Gemini or packages needed.

    python -m unittest discover -s tests -v
"""

import json
import unittest

from fovea.story.pieces import (AudioUnit, VideoUnit, av_segments, build_audio_units, build_video_units,
                                cut_pieces, pieces_table, spans_from_frames)

# The example from the brief, with an "aaa"-sized pause before the punchline that Whisper has no word for.
SHOTS = [(0.0, 4.0), (4.0, 8.0), (8.0, 11.0)]
WORDS = [
    {"start": 1.0, "end": 1.3, "word": "I"}, {"start": 1.35, "end": 1.7, "word": "hope"},
    {"start": 1.75, "end": 2.0, "word": "it's"}, {"start": 2.05, "end": 2.6, "word": "shoes."},
    {"start": 5.0, "end": 5.28, "word": "Oh"},                                   # then a 0.22 s "aaa"
    {"start": 5.5, "end": 5.7, "word": "A"}, {"start": 5.75, "end": 6.5, "word": "sweater."},
]
TALK = [(0.9, 2.7), (4.9, 6.6)]          # she is visibly talking; the rest of shots 0-1 is reaction


def tiles(units, total):
    """True when the units cover [0, total] with no holes and no overlaps."""
    edges = [(u.start, u.end) for u in units]
    return (abs(edges[0][0]) < 1e-6 and abs(edges[-1][1] - total) < 1e-6
            and all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(edges, edges[1:]))
            and all(e > s for s, e in edges))


class AudioUnitsTest(unittest.TestCase):
    def test_the_whole_video_is_covered_exactly_once(self):
        audio = build_audio_units(WORDS, SHOTS)
        self.assertTrue(tiles(audio, 11.0), [(a.start, a.end) for a in audio])
        self.assertEqual([a.id for a in audio], [f"A{i + 1:02d}" for i in range(len(audio))])

    def test_speech_is_split_at_pauses_and_punctuation_and_keeps_its_words_whole(self):
        speech = [a for a in build_audio_units(WORDS, SHOTS) if a.kind == "speech"]
        self.assertEqual([a.text for a in speech], ["I hope it's shoes.", "Oh", "A sweater."])
        for a in speech:
            self.assertLessEqual(a.start, a.words[0]["start"] + 1e-6)
            self.assertGreaterEqual(a.end, a.words[-1]["end"] - 1e-6)

    def test_a_short_audible_pause_is_its_own_unit_so_it_can_be_cut(self):
        audio = build_audio_units(WORDS, SHOTS)
        oh = next(i for i, a in enumerate(audio) if a.text == "Oh")
        gap, line = audio[oh + 1], audio[oh + 2]
        self.assertEqual((gap.kind, line.text), ("gap", "A sweater."))
        self.assertLess(gap.dur, 0.22)                 # the 0.22 s pause, minus a little air left on the words
        self.assertGreater(gap.dur, 0.08)
        self.assertGreater(gap.start, 5.28)
        self.assertLess(gap.end, 5.5)

    def test_a_pause_too_short_to_hear_stays_with_the_words(self):
        words = [{"start": 1.0, "end": 1.5, "word": "one"}, {"start": 1.6, "end": 2.0, "word": "two"}]
        audio = build_audio_units(words, [(0.0, 3.0)])
        self.assertEqual([a.text for a in audio if a.kind == "speech"], ["one two"])

    def test_a_phrase_running_across_a_shot_cut_stays_one_unit(self):
        words = [{"start": 3.6, "end": 3.95, "word": "hello"}, {"start": 3.98, "end": 4.4, "word": "there"}]
        audio, video = cut_pieces(SHOTS, words)
        phrase = next(a for a in audio if a.kind == "speech")
        self.assertEqual(phrase.text, "hello there")
        self.assertLess(phrase.start, 4.0)
        self.assertGreater(phrase.end, 4.0)
        self.assertEqual(len(phrase.over), 2)          # recorded over the pictures on both sides of the cut

    def test_silence_is_split_at_shot_cuts_and_into_short_chunks(self):
        audio = build_audio_units([], SHOTS)
        self.assertTrue(tiles(audio, 11.0))
        self.assertTrue(all(a.kind == "gap" and a.dur <= 2.5 + 1e-6 for a in audio))
        starts = {round(a.start, 3) for a in audio}
        self.assertTrue({4.0, 8.0} <= starts)          # a silent stretch can be dropped shot by shot

    def test_overlapping_or_broken_word_timings_still_give_clean_units(self):
        words = [{"start": 1.0, "end": 2.0, "word": "one."}, {"start": 1.5, "end": 2.5, "word": "two."},
                 {"start": 2.4, "end": 2.4, "word": "zero-length"}, {"start": 10.5, "end": 14.0, "word": "late"}]
        audio = build_audio_units(words, SHOTS)
        self.assertTrue(tiles(audio, 11.0), [(a.start, a.end) for a in audio])

    def test_no_shots_no_units(self):
        self.assertEqual(build_audio_units(WORDS, []), [])


class VideoUnitsTest(unittest.TestCase):
    def test_every_shot_is_one_video_unit(self):
        video = build_video_units(SHOTS, TALK)
        self.assertEqual([(v.id, v.start, v.end, v.shot) for v in video],
                         [("V01", 0.0, 4.0, 0), ("V02", 4.0, 8.0, 1), ("V03", 8.0, 11.0, 2)])

    def test_each_shot_carries_how_much_of_it_shows_someone_talking(self):
        video = build_video_units(SHOTS, TALK)
        self.assertEqual([v.talking for v in video], [0.45, 0.42, 0.0])      # the sweater close-up: nobody talks
        self.assertEqual(build_video_units([(0.0, 2.0)], [(0.0, 5.0)])[0].talking, 1.0)

    def test_detected_on_screen_text_is_carried(self):
        video = build_video_units(SHOTS, [], {2: ["For you"]})
        self.assertEqual([v.ocr for v in video], [[], [], ["For you"]])

    def test_frames_become_time_spans(self):
        self.assertEqual(spans_from_frames([25, 26, 27, 30, 100], 25.0), [(1.0, 1.24), (4.0, 4.04)])
        self.assertEqual(spans_from_frames([], 25.0), [])


class LinkAndSegmentsTest(unittest.TestCase):
    def test_every_sound_knows_the_picture_it_was_recorded_over(self):
        audio, video = cut_pieces(SHOTS, WORDS, TALK)
        ids = {v.id for v in video}
        for a in audio:
            self.assertTrue(a.over and set(a.over) <= ids, a)
        line = next(a for a in audio if a.text == "A sweater.")
        self.assertEqual(line.over, ["V02"])            # the punchline is recorded over shot 1

    def test_one_av_segment_per_audio_unit_covering_the_whole_video(self):
        audio, video = cut_pieces(SHOTS, WORDS, TALK)
        segs = av_segments(audio, video)
        self.assertEqual([s["audio"] for s in segs], [a.id for a in audio])
        self.assertEqual([s["id"] for s in segs], [f"S{i + 1:02d}" for i in range(len(segs))])
        self.assertAlmostEqual(sum(s["length"] for s in segs), 11.0, places=2)
        self.assertEqual(next(s for s in segs if s["audio"] == audio[0].id)["heard"], "(no speech)")

    def test_av_segments_use_the_call_1_descriptions_when_given(self):
        audio, video = cut_pieces(SHOTS, WORDS, TALK)
        line = next(a for a in audio if a.text == "A sweater.")
        segs = av_segments(audio, video, heard={line.id: "A sweater!"}, shows={line.over[0]: "her face falls"})
        seg = next(s for s in segs if s["audio"] == line.id)
        self.assertEqual((seg["heard"], seg["shows"]), ("A sweater!", ["her face falls"]))

    def test_pieces_survive_being_stored(self):
        audio, video = cut_pieces(SHOTS, WORDS, TALK, {2: ["For you"]})
        self.assertEqual([AudioUnit.from_dict(json.loads(json.dumps(a.to_dict()))) for a in audio], audio)
        self.assertEqual([VideoUnit.from_dict(json.loads(json.dumps(v.to_dict()))) for v in video], video)
        self.assertIn("visibly talking 45%", pieces_table(audio, video))


if __name__ == "__main__":
    unittest.main()
