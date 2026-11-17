"""Call 1 without Gemini: the prompt table, cleaning the answer, storing it.

    python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from fovea.story import describe as call1
from fovea.story.pieces import cut_pieces

from test_pieces import SHOTS, TALK, WORDS


def pieces():
    return cut_pieces(SHOTS, WORDS, TALK, {2: ["For yuo"]})


def answer(audio, video):
    """What a good answer for the sweater ad looks like."""
    line = next(a for a in audio if a.text == "A sweater.")
    setup = next(a for a in audio if a.text.startswith("I hope"))
    pause = audio[audio.index(line) - 1]
    return {
        "summary": "She hopes for shoes and gets a sweater.", "music": "bed",
        "video_units": [
            {"id": "V01", "shows": "a woman walks in with a gift", "focus": "her face", "needs": []},
            {"id": "V02", "shows": "her face falls as she opens it", "focus": "her face", "needs": ["V01"]},
            {"id": "V03", "shows": "a red sweater in the box; tag reads 'For you'", "focus": "the sweater",
             "needs": [line.id]},
        ],
        "audio_units": [{"id": a.id, "heard": {line.id: "A sweater.", pause.id: "(aaa)"}.get(a.id, a.text or "(silence)"),
                         "needs": [setup.id] if a.id == line.id else []} for a in audio],
    }


class Call1Test(unittest.TestCase):
    def test_the_prompt_lists_every_unit_with_its_measurements(self):
        audio, video = pieces()
        text = call1.prompt(audio, video)
        self.assertNotIn("{table}", text)
        for u in [*audio, *video]:
            self.assertIn(f"{u.id} | ", text)
        self.assertIn("visibly talking: 45%", text)
        self.assertIn("detected text: For yuo", text)
        self.assertIn('rough transcript: "A sweater."', text)
        self.assertIn("(no words transcribed)", text)

    def test_a_good_answer_passes_through_unchanged(self):
        audio, video = pieces()
        raw = answer(audio, video)
        described, issues = call1.clean(raw, audio, video)
        self.assertEqual(issues, [])
        self.assertEqual(described, raw)
        self.assertEqual(described["video_units"][1]["needs"], ["V01"])       # the reaction needs the setup shot

    def test_needs_may_only_point_at_earlier_known_units(self):
        audio, video = pieces()
        raw = answer(audio, video)
        raw["video_units"][0]["needs"] = ["V03", "V01", "Z9"]                 # later, itself, unknown
        raw["audio_units"][0]["needs"] = ["V01", "V01"]                       # same start time: allowed, once
        described, issues = call1.clean(raw, audio, video)
        self.assertEqual(described["video_units"][0]["needs"], [])
        self.assertEqual(described["audio_units"][0]["needs"], ["V01"])
        self.assertEqual(len(issues), 3)

    def test_forgotten_units_and_bad_values_are_filled_in(self):
        audio, video = pieces()
        raw = answer(audio, video)
        raw["video_units"] = raw["video_units"][:1] + [{"id": "V99", "shows": "?"}]
        raw["audio_units"] = [u for u in raw["audio_units"] if u["heard"] != "A sweater."]
        raw["music"] = "loud"
        described, issues = call1.clean(raw, audio, video)
        self.assertEqual([u["id"] for u in described["video_units"]], ["V01", "V02", "V03"])
        self.assertEqual(described["video_units"][2], {"id": "V03", "shows": "", "focus": "", "needs": []})
        line = next(a for a in audio if a.text == "A sweater.")
        got = next(u for u in described["audio_units"] if u["id"] == line.id)
        self.assertEqual(got["heard"], "A sweater.")                           # falls back to the rough transcript
        self.assertEqual(described["music"], "none")
        self.assertTrue(any("V99" in i for i in issues) and any("missing" in i for i in issues))

    def test_the_answer_is_stored_in_the_videos_folder_and_only_reused_for_that_file(self):
        audio, video = pieces()
        described, _ = call1.clean(answer(audio, video), audio, video)
        with tempfile.TemporaryDirectory() as d:
            prep = Path(d, "my_ad", "prep")
            self.assertIsNone(call1.load(prep, "abc"))
            call1.save(prep, "abc", audio, video, {"described": described, "issues": [], "model": "m", "seconds": 1.0})
            self.assertEqual(sorted(p.name for p in prep.iterdir()), ["call1.json", "pieces.json"])
            got_audio, got_video, record = call1.load(prep, "abc")
            self.assertEqual((got_audio, got_video, record["described"]), (audio, video, described))
            self.assertIsNone(call1.load(prep, "another-file"))
            (prep / "call1.json").write_text("{broken")
            self.assertIsNone(call1.load(prep, "abc"))

    def test_call_1_is_about_the_story_only_no_framing(self):
        audio, video = pieces()
        text = call1.prompt(audio, video)
        self.assertNotIn("GREEN", text)
        self.assertNotIn("focus_x", text)
        described, _ = call1.clean(answer(audio, video), audio, video)
        self.assertEqual(sorted(described["video_units"][0]), ["focus", "id", "needs", "shows"])

    def test_the_report_shows_what_gemini_saw_heard_and_linked(self):
        audio, video = pieces()
        described, _ = call1.clean(answer(audio, video), audio, video)
        text = call1.report(audio, video, described)
        self.assertIn("shows: her face falls as she opens it", text)
        self.assertIn("focus: the sweater", text)
        self.assertIn("heard: (aaa)", text)
        self.assertIn("needs V01", text)
        json.dumps(described)


if __name__ == "__main__":
    unittest.main()
