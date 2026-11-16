"""Call 2 without Gemini: the prompts, reading the answer, and turning an edit into exact pieces.

    python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest

from fovea.story import describe as call1
from fovea.story import edit_plan as call2

from test_describe import answer, pieces
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
        self.assertIn("V02 | 4.00s | shows: her face falls as she opens it | needs V01", text)
        self.assertIn(f"S{sweater[1:]} | 1.11s | over V02 | heard: A sweater. | needs S{hope[1:]}", text)
        self.assertNotIn(f"{sweater} |", text)                                  # no audio ids in the synced prompt

    def test_cutaway_prompt_gives_the_two_tracks_separately(self):
        audio, video, described, hope, sweater = setup()
        text = call2.cutaway_prompt(audio, video, described, [10])
        self.assertIn("V03 | 3.00s | visibly talking: 0% | shows: a red sweater in the box; tag reads 'For you' | "
                      f"focus: the sweater | needs {sweater}", text)
        self.assertIn(f"{sweater} | 1.11s | over V02 | heard: A sweater. | needs {hope}", text)


class ParseTest(unittest.TestCase):
    def test_edits_are_matched_to_the_requested_lengths(self):
        raw = {"edits": [{"target_s": 10, "segments": ["S02"], "story": "ten"},
                         {"target_s": 15, "segments": ["S02", "S07"], "story": "fifteen"}]}
        got = call2.parse(raw, [15, 10], "synced", "m")
        self.assertEqual(list(got), ["15s_synced", "10s_synced"])
        self.assertEqual(got["15s_synced"]["segments"], [{"audio": "A02", "video": ""}, {"audio": "A07", "video": ""}])
        self.assertEqual(got["10s_synced"]["story"], "ten")

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

    def test_a_shot_used_as_a_cover_is_not_also_shown_on_its_own(self):
        audio, video, described, hope, sweater = setup()
        own = next(a.id for a in audio if a.over == ["V03"])
        cut = call2.build_cut(audio, video, described, edit("cutaway", [hope, (sweater, "V03"), own]), TALK)
        self.assertFalse([p for p in cut["pieces"] if "<-" in p["units"]])       # the cover is undone
        self.assertTrue(any("also shown on its own" in n for n in cut["stats"]["notes"]))
        starts = [p["video_start"] for p in cut["pieces"]]
        self.assertEqual(len(starts), len(set(starts)))

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

    def test_bad_ids_and_repeats_are_dropped_and_reported(self):
        audio, video, described, hope, sweater = setup()
        cut = call2.build_cut(audio, video, described,
                              edit("cutaway", [hope, "A99", hope, (sweater, "V77")]), TALK)
        self.assertEqual(cut["stats"]["units"], [hope, sweater])
        notes = " | ".join(cut["stats"]["notes"])
        self.assertIn("'A99' ignored", notes)
        self.assertIn("listed twice", notes)
        self.assertIn("unknown picture 'V77'", notes)

    def test_a_cut_piece_that_a_kept_piece_needs_is_put_back(self):
        audio, video, described, hope, sweater = setup()
        cut = call2.build_cut(audio, video, described, edit("synced", [sweater], target=8.0), TALK)
        notes = " | ".join(cut["stats"]["notes"])
        self.assertIn(f"{sweater} needs {hope}, which was cut: put back ({hope})", notes)
        self.assertIn("under the 8s target", notes)                       # a missed length is reported, not fixed
        self.assertEqual(cut["stats"]["units"], [hope, sweater])          # in story order

    def test_an_ending_that_needs_a_cut_piece_brings_the_chain_back(self):
        audio, video, described, hope, sweater = setup()
        ending = [a.id for a in audio if a.over == ["V03"]]              # the sweater close-up needs her line...
        cut = call2.build_cut(audio, video, described, edit("synced", ending[:1], target=3.0), TALK)
        self.assertEqual(cut["stats"]["units"], [hope, sweater, ending[0]])   # ...and her line needs its setup
        self.assertEqual(len([n for n in cut["stats"]["notes"] if "put back" in n]), 2)

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
