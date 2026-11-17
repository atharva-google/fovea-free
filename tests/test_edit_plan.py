"""Call 2 without Gemini: the prompts, reading the answer, and turning an edit into exact pieces.

    python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest

from fovea.story import describe as call1
from fovea.story import edit_plan as call2

from test_describe import answer, pieces
from fovea.story.pieces import cut_pieces

from test_pieces import TALK


def setup():
    audio, video = pieces()
    described, _ = call1.clean(answer(audio, video), audio, video)
    ids = {a.text or f"gap{i}": a.id for i, a in enumerate(audio)}
    return audio, video, described, ids["I hope it's shoes."], ids["A sweater."]


def no_needs(described):
    for u in [*described["video_units"], *described["audio_units"]]:
        u["needs"] = []
    return described


def edit(mode, segments, target=5.0):
    return {"target_s": target, "mode": mode, "story": "s",
            "segments": [{"audio": s, "video": ""} if isinstance(s, str) else {"audio": s[0], "video": s[1]}
                         for s in segments]}


class PromptTest(unittest.TestCase):
    def test_synced_prompt_gives_segments_with_lengths_and_needs_as_segment_ids(self):
        audio, video, described, hope, sweater = setup()
        text = call2.synced_prompt(audio, video, described, [15, 10])
        self.assertNotIn("{", text.split("SHOTS (what is seen)")[0])           # every placeholder was filled
        self.assertIn("each of these lengths: 15s, 10s".lower(), text.lower())
        self.assertIn("Full length: 11.0s", text)
        self.assertIn("V02 | 4.00s | cast: P1 | shows: her face falls as she opens it | needs V01", text)
        self.assertIn(f"S{sweater[1:]} | 1.11s | over V02 | heard: P1: A sweater. | needs S{hope[1:]}", text)
        self.assertNotIn(f"{sweater} |", text)                                  # no audio ids in the synced prompt

    def test_cutaway_prompt_gives_the_two_tracks_separately(self):
        audio, video, described, hope, sweater = setup()
        text = call2.cutaway_prompt(audio, video, described, [10], TALK)
        self.assertIn(f"V03 | 3.00s | CAN COVER up to 3.0s | shows: a red sweater in the box | needs {sweater}", text)
        self.assertIn("P1 (Mira): woman, about 30, red kurta", text)
        self.assertIn(f"{sweater} | 1.11s | own picture V02 | heard: P1: A sweater. | needs {hope}", text)
        self.assertIn("V01 | 4.00s | CAN COVER up to 1.3s |", text)             # only her silent reaction
        busy = call2.cutaway_prompt(audio, video, described, [10], [(0.0, 3.8)])
        self.assertIn("V01 | 4.00s | cannot cover (a face is talking) |", busy)

    def test_a_short_shot_with_nobody_talking_can_cover_a_short_sound(self):
        # from a real ad: a 0.68 s product close-up was labelled "cannot cover (a face is talking)"
        shots = [(0.0, 3.0), (3.0, 3.68), (3.68, 3.9), (3.9, 7.0)]
        words = [{"start": 0.5, "end": 0.9, "word": "Look."}, {"start": 4.5, "end": 6.5, "word": "longer line"}]
        audio, video = cut_pieces(shots, words)
        described, _ = call1.clean({}, audio, video)
        text = call2.cutaway_prompt(audio, video, described, [4])
        self.assertIn("V02 | 0.68s | CAN COVER up to 0.7s |", text)
        self.assertIn("V03 | 0.22s | cannot cover (too short) |", text)
        look = next(a.id for a in audio if a.text == "Look.")
        long = next(a.id for a in audio if a.text == "longer line")
        cut = call2.build_cut(audio, video, described, edit("cutaway", [(look, "V02")], target=1.0))
        self.assertEqual(cut["stats"]["covers"], {look: "V02"})                 # ~0.5 s of sound fits in 0.68 s

    def test_a_cover_shorter_than_its_sound_is_shown_first_then_the_sounds_own_picture(self):
        # from a real ad: a 0.68 s close-up of the pack over a ~1 s line that names it
        shots = [(0.0, 3.0), (3.0, 3.68), (3.68, 7.0)]
        words = [{"start": 4.5, "end": 5.4, "word": "Hotel Kohinoor."}]
        audio, video = cut_pieces(shots, words)
        described, _ = call1.clean({}, audio, video)
        line = next(a for a in audio if a.text)
        cut = call2.build_cut(audio, video, described, edit("cutaway", [(line.id, "V02")], target=1.0))
        self.assertEqual(cut["stats"]["covers"], {line.id: "V02"})
        cover, own = cut["pieces"]
        self.assertEqual((cover["units"], cover["video_start"], cover["length"]), (f"{line.id}<-V02", 3.0, 0.68))
        self.assertAlmostEqual(cover["audio_start"], line.start, places=3)        # the pack as the line starts
        self.assertAlmostEqual(own["video_start"], line.start + 0.68, places=3)   # then the speaker, in sync
        self.assertEqual(own["video_start"], own["audio_start"])
        self.assertAlmostEqual(cover["length"] + own["length"], line.dur, places=3)

    def test_the_tail_left_on_its_own_picture_is_not_reused_as_a_cover(self):
        shots = [(0.0, 0.7), (0.7, 4.0), (4.0, 8.0)]
        words = [{"start": 1.0, "end": 3.6, "word": "a long line."}, {"start": 5.0, "end": 6.0, "word": "later."}]
        audio, video = cut_pieces(shots, words)
        described, _ = call1.clean({}, audio, video)
        first = next(a.id for a in audio if a.text == "a long line.")
        later = next(a.id for a in audio if a.text == "later.")
        cut = call2.build_cut(audio, video, described, edit("cutaway", [(first, "V01"), (later, "V02")], target=4.0))
        seen = sorted((p["video_start"], p["video_start"] + p["length"]) for p in cut["pieces"])
        for a, b in zip(seen, seen[1:]):
            self.assertGreaterEqual(b[0], a[1] - 1e-6)                            # no footage twice


# The case from a real ad: a closing voiceover runs, without a pause, over the last two shots.
VO_SHOTS = [(0.0, 4.0), (4.0, 8.0), (8.0, 11.0)]
VO_WORDS = [{"start": 1.0, "end": 2.0, "word": "Skincare!"},
            *[{"start": 6.0 + 0.5 * i, "end": 6.5 + 0.5 * i, "word": f"w{i}"} for i in range(10)]]   # 6.0 - 11.0


def voiceover():
    audio, video = cut_pieces(VO_SHOTS, VO_WORDS)
    described, _ = call1.clean({}, audio, video)
    line = next(a.id for a in audio if a.text == "Skincare!")
    vo = next(a.id for a in audio if a.text.startswith("w0"))
    return audio, video, described, line, vo


class SlotsTest(unittest.TestCase):
    def test_a_sound_running_over_two_shots_is_one_slot_per_shot(self):
        audio, video, described, line, vo = voiceover()
        slots = {s["id"]: s for s in call2.make_slots(audio, video)}
        self.assertIn(line, slots)                                    # over one shot: keeps its id
        self.assertEqual((slots[vo + "a"]["shot"], slots[vo + "b"]["shot"]), ("V02", "V03"))
        self.assertAlmostEqual(slots[vo + "a"]["end"], 8.0)
        self.assertAlmostEqual(slots[vo + "b"]["start"], 8.0)
        text = call2.cutaway_prompt(audio, video, described, [6])
        self.assertIn(f"{vo}b | 3.00s | own picture V03 | heard: (the same sound continues)", text)

    def test_a_sliver_of_sound_past_a_cut_does_not_become_a_slot(self):
        words = [{"start": 3.0, "end": 4.1, "word": "almost"}]
        audio, video = cut_pieces(VO_SHOTS, words)
        ids = [s["id"] for s in call2.make_slots(audio, video)]
        self.assertFalse([i for i in ids if i.endswith(("a", "b"))])

    def test_borrowing_a_shot_that_the_voiceover_still_shows_is_refused_and_explained(self):
        audio, video, described, line, vo = voiceover()
        cut = call2.build_cut(audio, video, described, edit("cutaway", [(line, "V03"), vo]))
        self.assertFalse([p for p in cut["pieces"] if "<-" in p["units"]])
        self.assertTrue(any(f"V03 cannot cover {line}: it is already shown under {vo}b" in n
                            for n in cut["stats"]["notes"]), cut["stats"]["notes"])

    def test_re_covering_the_voiceover_slot_makes_the_cutaway_work(self):
        audio, video, described, line, vo = voiceover()
        cut = call2.build_cut(audio, video, described,
                              edit("cutaway", [(line, "V03"), (vo + "a", "V02"), (vo + "b", "V01")]))
        self.assertEqual(cut["stats"]["covers"], {line: "V03", vo + "b": "V01"})
        self.assertFalse(any("cannot cover" in n or n.startswith("NO CUTAWAYS") for n in cut["stats"]["notes"]))
        seen = sorted((p["video_start"], p["video_start"] + p["length"]) for p in cut["pieces"])
        for a, b in zip(seen, seen[1:]):
            self.assertGreaterEqual(b[0], a[1] - 1e-6)                # no footage twice
        heard = sorted((p["audio_start"], p["audio_start"] + p["length"]) for p in cut["pieces"])
        self.assertAlmostEqual(heard[-1][1], 11.0, places=2)          # the voiceover is heard to its end
        self.assertAlmostEqual(heard[-2][1], heard[-1][0], places=2)  # ...without a break at the shot cut

    def test_a_sound_is_never_split_and_a_whole_unit_can_be_named(self):
        audio, video, described, line, vo = voiceover()
        cut = call2.build_cut(audio, video, described, edit("cutaway", [(vo + "b", "V01")]))
        self.assertEqual(cut["stats"]["units"], [vo + "a", vo + "b"])
        self.assertTrue(any("one sound" in n for n in cut["stats"]["notes"]))
        cut = call2.build_cut(audio, video, described, edit("synced", [vo]))
        self.assertEqual(cut["stats"]["units"], [vo + "a", vo + "b"])
        self.assertEqual(len(cut["pieces"]), 1)                       # synced: one continuous piece


class ParseTest(unittest.TestCase):
    def test_edits_are_matched_to_the_requested_lengths(self):
        raw = {"edits": [{"target_s": 10, "segments": ["S02"], "story": "ten"},
                         {"target_s": 15, "segments": ["S02", "S07"], "story": "fifteen"}]}
        got = call2.parse(raw, [15, 10], "synced", "m")
        self.assertEqual(list(got), ["15s_synced", "10s_synced"])
        self.assertEqual(got["15s_synced"]["segments"], [{"audio": "A02", "video": ""}, {"audio": "A07", "video": ""}])
        self.assertEqual(got["10s_synced"]["story"], "ten")

    def test_the_story_score_is_read_and_shown(self):
        raw = {"edits": [{"target_s": 10, "segments": ["S02"], "story": "s", "story_score": 4.4,
                          "score_reason": "the payoff has no setup"},
                         {"target_s": 15, "segments": ["S02"], "story": "s", "story_score": 99}]}
        got = call2.parse(raw, [10, 15, 20], "synced")
        self.assertEqual((got["10s_synced"]["story_score"], got["10s_synced"]["score_reason"]), (4, "the payoff has no setup"))
        self.assertEqual(got["15s_synced"]["story_score"], 10)
        self.assertIsNone(call2.parse({"edits": [{"target_s": 10, "segments": []}]}, [10], "synced")["10s_synced"]["story_score"])
        audio, video, described, hope, sweater = setup()
        cut = call2.build_cut(audio, video, described, {**edit("synced", [hope, sweater]), **got["10s_synced"],
                                                        "segments": [{"audio": hope, "video": ""}]})
        self.assertIn("story score: 4/10  LOW  the payoff has no setup", call2.report({"x": cut}))

    def test_cutaway_segments_and_a_missing_edit(self):
        raw = {"edits": [{"target_s": 10, "segments": [{"audio": "A07", "video": "V03"}], "story": ""}]}
        got = call2.parse(raw, [10, 6], "cutaway")
        self.assertEqual(list(got), ["10s_cutaway"])
        self.assertEqual(got["10s_cutaway"]["segments"], [{"audio": "A07", "video": "V03"}])

    def test_edits_are_stored_for_one_file_only(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(call2.load(d, "abc"), {})
            call2.save(d, "abc", {"10s_synced": edit("synced", ["A02"])})
            self.assertEqual(list(call2.load(d, "abc")), ["10s_synced"])
            self.assertEqual(call2.load(d, "other"), {})


class BuildCutTest(unittest.TestCase):
    def test_synced_keeps_each_sound_with_its_own_picture(self):
        audio, video, described, hope, sweater = setup()
        cut = call2.build_cut(audio, video, described, edit("synced", [hope, sweater]), TALK)
        self.assertTrue(cut["stats"]["ok"])
        self.assertEqual(len(cut["pieces"]), 2)
        for p in cut["pieces"]:
            self.assertEqual(p["video_start"], p["audio_start"])
        self.assertAlmostEqual(cut["stats"]["total"], 1.72 + 1.11, places=1)
        json.dumps(cut)

    def test_neighbouring_segments_become_one_piece(self):
        audio, video, described, *_ = setup()
        cut = call2.build_cut(audio, video, described, edit("synced", [a.id for a in audio[1:4]]), TALK)
        self.assertEqual(len(cut["pieces"]), 1)
        self.assertEqual(cut["pieces"][0]["units"], "A02 A03 A04")

    def test_dropping_the_aaa_closes_the_gap(self):
        audio, video, described, hope, sweater = setup()
        i = next(k for k, a in enumerate(audio) if a.id == sweater)
        oh, aaa = audio[i - 2], audio[i - 1]
        no_needs(described)
        with_it = call2.build_cut(audio, video, described, edit("synced", [oh.id, aaa.id, sweater]), TALK)
        without = call2.build_cut(audio, video, described, edit("synced", [oh.id, sweater]), TALK)
        self.assertEqual((len(with_it["pieces"]), len(without["pieces"])), (1, 2))
        self.assertAlmostEqual(with_it["stats"]["total"] - without["stats"]["total"], aaa.dur, places=2)

    def test_cutaway_shows_the_sweater_while_the_line_plays(self):
        audio, video, described, hope, sweater = setup()
        cut = call2.build_cut(audio, video, described, edit("cutaway", [hope, (sweater, "V03")], target=3.0), TALK)
        over = [p for p in cut["pieces"] if "<-" in p["units"]]
        self.assertEqual([p["units"] for p in over], [f"{sweater}<-V03"])
        self.assertEqual(over[0]["video_start"], 8.0)                           # the sweater shot
        self.assertAlmostEqual(over[0]["audio_start"], 5.45, places=1)          # her line
        self.assertEqual(cut["stats"]["covers"], {sweater: "V03"})
        self.assertEqual(cut["stats"]["notes"], [])

    def test_a_shot_can_cover_and_be_shown_but_no_footage_is_seen_twice(self):
        audio, video, described, hope, sweater = setup()
        own = next(a for a in audio if a.over == ["V03"])                       # first 1.5 s of the close-up
        cut = call2.build_cut(audio, video, described, edit("cutaway", [hope, (sweater, "V03"), own.id]), TALK)
        over = [p for p in cut["pieces"] if "<-" in p["units"]]
        self.assertEqual(len(over), 1)
        self.assertGreaterEqual(over[0]["video_start"], own.end - 1e-6)         # the cover uses the rest of the shot
        seen = sorted((p["video_start"], p["video_start"] + p["length"]) for p in cut["pieces"])
        for a, b in zip(seen, seen[1:]):
            self.assertGreaterEqual(b[0], a[1] - 1e-6)

    def test_a_cover_is_undone_when_all_of_the_shot_is_shown_on_its_own(self):
        audio, video, described, hope, sweater = setup()
        own = [a.id for a in audio if a.over == ["V03"]]                        # the whole close-up
        cut = call2.build_cut(audio, video, described, edit("cutaway", [hope, (sweater, "V03"), *own]), TALK)
        self.assertFalse([p for p in cut["pieces"] if "<-" in p["units"]])
        self.assertTrue(any("V03 cannot cover" in n and "already shown under" in n for n in cut["stats"]["notes"]))

    def test_a_cover_continues_and_never_repeats_footage(self):
        audio, video, described, hope, sweater = setup()
        oh = audio[next(k for k, a in enumerate(audio) if a.id == sweater) - 2].id
        cut = call2.build_cut(audio, video, described, edit("cutaway", [(hope, "V03"), (oh, "V03"), (sweater, "V03")]), TALK)
        over = [p for p in cut["pieces"] if "<-" in p["units"]]
        spans = sorted((p["video_start"], p["video_start"] + p["length"]) for p in over)
        self.assertEqual(spans[0][0], 8.0)
        for a, b in zip(spans, spans[1:]):
            self.assertGreaterEqual(b[0], a[1] - 1e-6)                          # each use starts where the last ended
        self.assertLessEqual(spans[-1][1], 11.0 + 1e-6)

    def test_a_cover_avoids_the_part_of_a_shot_where_a_face_is_talking(self):
        audio, video, described, hope, sweater = setup()
        # cover her punchline with shot V01, where she talks from 0.9 to 2.7 s: the picture must come from after that
        cut = call2.build_cut(audio, video, no_needs(described), edit("cutaway", [(sweater, "V01")]), TALK)
        over = [p for p in cut["pieces"] if "<-" in p["units"]]
        self.assertEqual(len(over), 1)
        self.assertGreaterEqual(over[0]["video_start"], 2.7)
        self.assertLessEqual(over[0]["video_start"] + over[0]["length"], 4.0 + 1e-6)

    def test_a_shot_with_no_quiet_footage_cannot_cover(self):
        audio, video, described, hope, sweater = setup()
        cut = call2.build_cut(audio, video, no_needs(described), edit("cutaway", [(sweater, "V01")]), [(0.0, 4.0)])
        self.assertFalse([p for p in cut["pieces"] if "<-" in p["units"]])
        self.assertTrue(any("without a talking face" in n for n in cut["stats"]["notes"]))
        self.assertTrue(any(n.startswith("NO CUTAWAYS") for n in cut["stats"]["notes"]))

    def test_bad_ids_and_repeats_are_dropped_and_reported(self):
        audio, video, described, hope, sweater = setup()
        cut = call2.build_cut(audio, video, described,
                              edit("cutaway", [hope, "A99", hope, (sweater, "V77")]), TALK)
        self.assertEqual(cut["stats"]["units"], [hope, sweater])
        notes = " | ".join(cut["stats"]["notes"])
        self.assertIn("'A99' ignored", notes)
        self.assertIn("listed twice", notes)
        self.assertIn("unknown picture 'V77'", notes)

    def test_an_unmet_need_and_a_missed_target_are_reported_not_changed(self):
        audio, video, described, hope, sweater = setup()
        cut = call2.build_cut(audio, video, described, edit("synced", [sweater], target=8.0), TALK)
        notes = " | ".join(cut["stats"]["notes"])
        self.assertIn(f"{sweater} needs {hope}", notes)
        self.assertIn("under the 8s target", notes)
        self.assertEqual(cut["stats"]["units"], [sweater])                # Gemini's edit is rendered as it asked

    def test_a_cutaway_edit_without_any_cutaway_says_so(self):
        audio, video, described, hope, sweater = setup()
        cut = call2.build_cut(audio, video, described, edit("cutaway", [(hope, "V01"), (sweater, "V02")]), TALK)
        self.assertTrue(any(n.startswith("NO CUTAWAYS") for n in cut["stats"]["notes"]))
        cut = call2.build_cut(audio, video, described, edit("cutaway", [hope, (sweater, "V03")]), TALK)
        self.assertFalse(any(n.startswith("NO CUTAWAYS") for n in cut["stats"]["notes"]))

    def test_an_edit_that_keeps_everything_or_nothing_is_not_usable(self):
        audio, video, described, *_ = setup()
        self.assertFalse(call2.build_cut(audio, video, described, edit("synced", [a.id for a in audio]), TALK)["stats"]["ok"])
        self.assertFalse(call2.build_cut(audio, video, described, edit("synced", []), TALK)["stats"]["ok"])

    def test_the_report_shows_gemini_s_reason_and_the_pieces(self):
        audio, video, described, hope, sweater = setup()
        cut = call2.build_cut(audio, video, described, edit("cutaway", [hope, (sweater, "V03")]), TALK)
        text = call2.report({"5s_cutaway_gemini": cut})
        self.assertIn("=== 5s_cutaway_gemini:", text)
        self.assertIn("Gemini: s", text)
        self.assertIn("(sound from 5.4", text)


if __name__ == "__main__":
    unittest.main()
