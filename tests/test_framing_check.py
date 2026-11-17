"""The optional framing check, without Gemini: prompt, reading the answer, storage.

    python -m unittest discover -s tests -v
"""

import tempfile
import unittest

from fovea.story import framing_check as check
from fovea.story.pieces import build_video_units

VIDEO = build_video_units([(0.0, 4.0), (4.0, 8.0), (8.0, 11.0)])


class FramingCheckTest(unittest.TestCase):
    def test_the_prompt_lists_the_shots_and_explains_the_line(self):
        text = check.prompt(VIDEO)
        self.assertIn("GREEN VERTICAL LINE", text)
        self.assertIn("V02 | 4.00-8.00s (4.00s)", text)
        self.assertNotIn("{shots}", text)

    def test_only_corrected_shots_become_fixes(self):
        raw = {"shots": [{"id": "V01", "fixed_focus_x": -1, "why": ""},
                         {"id": "V03", "fixed_focus_x": 55, "why": "the line is on the logo; the bars are in the middle"}]}
        shots = check.clean(raw, VIDEO)
        self.assertEqual([s["fixed_focus_x"] for s in shots], [-1, -1, 55])          # V02 was left out: no fix
        self.assertEqual(check.focus_fixes(VIDEO, shots), [(8.0, 11.0, 55)])
        self.assertIn("V03 8.0-11.0s -> centre at 55%", check.report(VIDEO, shots))
        self.assertIn("1 of 3 shots", check.report(VIDEO, shots))

    def test_bad_values_mean_no_fix(self):
        raw = {"shots": [{"id": "V01", "fixed_focus_x": 250}, {"id": "V02", "fixed_focus_x": "left"},
                         {"id": "V09", "fixed_focus_x": 40}]}
        self.assertEqual(check.focus_fixes(VIDEO, check.clean(raw, VIDEO)), [])
        self.assertEqual(check.focus_fixes(VIDEO, check.clean({}, VIDEO)), [])

    def test_the_answer_is_reused_only_for_the_same_file_and_the_same_line(self):
        line = check.signature([960.0, 960.4, 1200.0])
        self.assertEqual(line, check.signature([960.2, 960.0, 1200.3]))               # same to the pixel
        self.assertNotEqual(line, check.signature([960.0, 960.0, 1300.0]))
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(check.load(d, "abc", line))
            check.save(d, "abc", line, {"shots": check.clean({}, VIDEO), "model": "m", "seconds": 1.0})
            self.assertEqual(len(check.load(d, "abc", line)["shots"]), 3)
            self.assertIsNone(check.load(d, "another-file", line))
            self.assertIsNone(check.load(d, "abc", "another-line"))                   # the framing changed


if __name__ == "__main__":
    unittest.main()
