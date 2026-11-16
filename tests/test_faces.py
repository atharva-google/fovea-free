"""Faces on the whole frame -> merged, given a margin, linked into tracks (fovea/models/faces.py).

    python -m unittest discover -s tests -v
"""

import importlib.util
import unittest
from pathlib import Path

# loaded by path: importing the ``fovea.models`` package pulls in nothing heavy, but keep this test
# independent of the packages the detectors need
_spec = importlib.util.spec_from_file_location("faces", Path(__file__).resolve().parent.parent / "fovea/models/faces.py")
faces = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(faces)

W, H = 1280, 720


def face(x, y, size, score=0.9, sharp=100.0):
    return ([x, y, x + size, y + size], score, sharp)


def body(x1, y1, x2, y2, conf=0.8, cname="body"):
    return {"bbox": [x1, y1, x2, y2], "conf": conf, "cname": cname}


class MergeAndMarginTest(unittest.TestCase):
    def test_the_same_face_from_two_models_is_kept_once_with_the_better_score(self):
        got = faces.merge_faces([face(400, 100, 240, 0.7), face(410, 110, 230, 0.9), face(900, 100, 120, 0.8)])
        self.assertEqual([round(s, 1) for _, s, _ in got], [0.9, 0.8])

    def test_a_small_box_inside_a_big_one_is_the_same_face(self):
        got = faces.merge_faces([face(300, 50, 600, 0.6), face(500, 250, 150, 0.8)])   # a close-up found at two sizes
        self.assertEqual(len(got), 1)

    def test_the_margin_grows_the_box_in_proportion_and_stays_in_the_frame(self):
        self.assertEqual(faces.expand_box([400, 200, 600, 400], 0.15, W, H), [370, 170, 630, 430])
        self.assertEqual(faces.expand_box([0, 0, 200, 200], 0.15, W, H), [0, 0, 230, 230])
        self.assertEqual(faces.expand_box([1100, 600, 1280, 720], 0.5, W, H)[2:], [W, H])


class LinkerTest(unittest.TestCase):
    def test_a_face_keeps_its_id_while_it_moves_a_little(self):
        link = faces.Linker()
        ids = [link.update(f, [[400 + 4 * f, 100, 640 + 4 * f, 340]])[0] for f in range(0, 40, 2)]
        self.assertEqual(set(ids), {1})

    def test_two_faces_keep_their_own_ids_whatever_order_they_come_in(self):
        link = faces.Linker()
        a, b = [100, 100, 300, 300], [800, 100, 1000, 300]
        self.assertEqual(link.update(0, [a, b]), [1, 2])
        self.assertEqual(link.update(2, [b, a]), [2, 1])

    def test_a_short_dropout_keeps_the_id_and_a_long_one_starts_a_new_track(self):
        link = faces.Linker(max_age=12)
        box = [400, 100, 640, 340]
        self.assertEqual(link.update(0, [box]), [1])
        self.assertEqual(link.update(10, [box]), [1])
        self.assertEqual(link.update(40, [box]), [2])

    def test_a_new_face_elsewhere_gets_a_new_id(self):
        link = faces.Linker()
        link.update(0, [[100, 100, 300, 300]])
        self.assertEqual(link.update(2, [[100, 100, 300, 300], [900, 400, 1000, 500]]), [1, 2])


class AssembleTest(unittest.TestCase):
    def test_a_close_up_face_with_no_person_box_is_a_person(self):
        # the missed shot from a real ad: YOLO sees no body, the face models see the face
        frames = {f: {"faces": [face(300, 40, 620)], "yolo": []} for f in range(531, 557, 2)}
        out = faces.assemble_shot(frames, W, H)
        for f, dets in out.items():
            self.assertEqual([(d["cname"], d["track_id"]) for d in dets], [("person", 1)])
            self.assertEqual(dets[0]["bbox"], dets[0]["face_bbox"])           # no body: the face is the box
            self.assertGreater(faces.area(dets[0]["face_bbox"]), 620 * 620)   # margin added

    def test_a_face_inside_a_person_box_takes_that_box_as_its_body(self):
        frames = {0: {"faces": [face(500, 100, 200)], "yolo": [body(350, 50, 900, 720, 0.85)]}}
        d = faces.assemble_shot(frames, W, H)[0][0]
        self.assertEqual((d["cname"], d["bbox"], d["conf"]), ("person", [350, 50, 900, 720], 0.85))
        self.assertEqual(d["face_conf"], 0.9)
        self.assertEqual(d["sharp"], 100.0)

    def test_a_person_without_a_face_is_kept_as_a_body_and_objects_keep_their_class(self):
        frames = {0: {"faces": [], "yolo": [body(100, 100, 400, 700), body(186, 24, 1062, 690, 0.6, "cell phone")]},
                  2: {"faces": [], "yolo": [body(104, 100, 404, 700), body(190, 24, 1066, 690, 0.7, "toothbrush")]}}
        out = faces.assemble_shot(frames, W, H)
        self.assertEqual([(d["cname"], d["track_id"]) for d in out[0]], [("body", 1000), ("cell phone", 2000)])
        self.assertEqual([(d["cname"], d["track_id"]) for d in out[2]], [("body", 1000), ("toothbrush", 2000)])
        self.assertNotIn("face_bbox", out[0][0])

    def test_two_faces_in_one_body_box_do_not_both_take_it(self):
        frames = {0: {"faces": [face(400, 100, 150), face(700, 120, 140, 0.8)], "yolo": [body(300, 50, 1000, 720)]}}
        out = faces.assemble_shot(frames, W, H)[0]
        self.assertEqual([d["cname"] for d in out], ["person", "person"])
        self.assertEqual(sum(d["bbox"] == [300, 50, 1000, 720] for d in out), 1)

    def test_ids_are_stable_over_the_shot_and_frames_without_anything_are_empty(self):
        frames = {f: {"faces": [face(400 + f, 100, 200)] if f != 4 else [], "yolo": []} for f in range(0, 12, 2)}
        out = faces.assemble_shot(frames, W, H)
        self.assertEqual(out[4], [])
        self.assertEqual({d["track_id"] for f in out for d in out[f]}, {1})


def person(tid, bbox, face_box):
    return {"track_id": tid, "cname": "person", "bbox": bbox, "conf": 0.8, "face_bbox": face_box, "face_conf": 0.9}


class AddMissedFacesTest(unittest.TestCase):
    def test_what_was_already_found_is_left_exactly_as_it_was(self):
        known = person(3, [350, 50, 900, 720], [500, 100, 700, 300])
        frames = {0: {"dets": [known], "full": [face(505, 105, 190, 0.95)]}}     # the same face, seen again
        self.assertEqual(faces.add_missed_faces(frames)[0], [known])

    def test_a_close_up_with_no_person_box_is_added_and_keeps_one_id_over_the_shot(self):
        frames = {f: {"dets": [], "full": [face(300 + f, 40, 620)]} for f in range(531, 557, 2)}
        out = faces.add_missed_faces(frames)
        ids = {d["track_id"] for dets in out.values() for d in dets}
        self.assertEqual(ids, {faces.MISSED_ID0})
        d = out[531][0]
        self.assertEqual((d["cname"], d["bbox"], d["face_bbox"]), ("person", [300 + 531, 40, 920 + 531, 660], [300 + 531, 40, 920 + 531, 660]))

    def test_a_person_whose_face_was_missed_in_their_box_gets_it_and_keeps_their_track(self):
        missed = {"track_id": 3, "cname": "body", "bbox": [0, 23, 894, 696], "conf": 0.76}
        other = person(1, [0, 121, 238, 647], [20, 260, 180, 430])
        frames = {0: {"dets": [other, missed], "full": [face(330, 60, 500), face(25, 265, 150)]}}
        out = faces.add_missed_faces(frames)[0]
        self.assertEqual([(d["track_id"], d["cname"]) for d in out], [(1, "person"), (3, "person")])
        self.assertEqual(out[1]["face_bbox"], [330, 60, 830, 560])
        self.assertEqual(out[1]["bbox"], [0, 23, 894, 696])
        self.assertEqual(out[0], other)

    def test_bodies_and_objects_without_a_new_face_pass_through(self):
        dets = [{"track_id": 7, "cname": "body", "bbox": [100, 100, 300, 700], "conf": 0.6},
                {"track_id": 9, "cname": "cell phone", "bbox": [186, 24, 1062, 690], "conf": 0.7}]
        self.assertEqual(faces.add_missed_faces({0: {"dets": dets, "full": []}})[0], dets)

    def test_the_input_is_not_changed(self):
        missed = {"track_id": 3, "cname": "body", "bbox": [0, 0, 900, 700], "conf": 0.7}
        frames = {0: {"dets": [missed], "full": [face(300, 60, 500)]}}
        faces.add_missed_faces(frames)
        self.assertEqual(missed["cname"], "body")


if __name__ == "__main__":
    unittest.main()
