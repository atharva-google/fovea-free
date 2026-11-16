"""Faces found on the whole frame, linked into tracks by our own simple linker (no ByteTrack).

Plain Python (no numpy / mediapipe), tested in tests/test_faces.py. The models are run in
``tracker.py``; this file turns their raw boxes into the detections the rest of the pipeline uses.

Per frame (``assemble_shot``):
  * faces from the two MediaPipe models (short-range for close-ups, full-range for the rest) are
    merged: the same face found twice is kept once
  * each face box gets a margin, because MediaPipe boxes are tight (inside the face) and TalkNet
    needs the whole face with the chin
  * faces are linked from frame to frame into tracks, so that a face is "the same person" over a
    shot (TalkNet needs a continuous face track; the framing needs to know who it is following)
  * a face inside a YOLO person box takes that box as its body
  * YOLO person boxes WITHOUT a face are kept as ``body`` (a fallback for the framing, used only
    when no face is there), and everything else YOLO finds as objects

Output, per frame, a list of detections like the old tracker's::

    {"track_id", "cname": "person", "bbox", "conf", "face_bbox", "face_conf", "sharp"}   a face
    {"track_id", "cname": "body",   "bbox", "conf"}                                      a person, no face
    {"track_id", "cname": "<yolo class>", "bbox", "conf"}                                an object
"""

BODY_ID0, OBJECT_ID0 = 1000, 2000     # track ids: faces 1.., bodies 1000.., objects 2000..


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def merge_faces(faces, same=0.3):
    """``faces``: ``[(box, score, sharpness)]`` from one or more detectors. Two boxes are the same face
    when they overlap enough or one's centre lies inside the other; the higher score is kept."""
    kept = []
    for f in sorted(faces, key=lambda f: -f[1]):
        if not any(iou(f[0], k[0]) >= same or _inside(_centre(f[0]), k[0]) or _inside(_centre(k[0]), f[0])
                   for k in kept):
            kept.append(f)
    return kept


def expand_box(box, margin, frame_w, frame_h):
    """Grow a box by ``margin`` of its own width / height on every side, inside the frame."""
    w, h = box[2] - box[0], box[3] - box[1]
    return [int(max(0, round(box[0] - margin * w))), int(max(0, round(box[1] - margin * h))),
            int(min(frame_w, round(box[2] + margin * w))), int(min(frame_h, round(box[3] + margin * h)))]


class Linker:
    """Gives boxes the same id from frame to frame when they overlap the box that id had last.

    A track that is not seen for more than ``max_age`` frames is closed. Good enough inside one
    shot, where things move little between frames; two boxes that cross can swap ids."""

    def __init__(self, first_id=1, min_iou=0.2, max_age=12, grow=0.2):
        self.next_id, self.min_iou, self.max_age, self.grow = first_id, min_iou, max_age, grow
        self.tracks = {}                  # id -> (box, last frame)

    def _grown(self, b):
        w, h = (b[2] - b[0]) * self.grow, (b[3] - b[1]) * self.grow
        return [b[0] - w, b[1] - h, b[2] + w, b[3] + h]

    def update(self, frame, boxes):
        """Ids for ``boxes`` on this frame (frames must come in order)."""
        self.tracks = {i: t for i, t in self.tracks.items() if frame - t[1] <= self.max_age}
        pairs = sorted(((iou(self._grown(b), self._grown(t[0])), k, i)
                        for k, b in enumerate(boxes) for i, t in self.tracks.items()), reverse=True)
        ids, taken = [None] * len(boxes), set()
        for score, k, i in pairs:
            if score < self.min_iou:
                break
            if ids[k] is None and i not in taken:
                ids[k] = i
                taken.add(i)
        for k, b in enumerate(boxes):
            if ids[k] is None:
                ids[k] = self.next_id
                self.next_id += 1
            self.tracks[ids[k]] = (list(b), frame)
        return ids


def assemble_shot(frames, frame_w, frame_h, margin=0.15):
    """``frames``: ``{frame_idx: {"faces": [(box, score, sharpness)], "yolo": [{"bbox", "conf", "cname"}]}}``
    for ONE shot (YOLO's person class named ``"body"``). Returns ``{frame_idx: [detections]}``."""
    faces_l, bodies_l, objects_l = Linker(1), Linker(BODY_ID0), Linker(OBJECT_ID0)
    out = {}
    for f in sorted(frames):
        yolo = frames[f].get("yolo") or []
        bodies = [d for d in yolo if d["cname"] == "body"]
        objects = [d for d in yolo if d["cname"] != "body"]
        faces = [(expand_box(b, margin, frame_w, frame_h), s, sh) for b, s, sh in merge_faces(frames[f].get("faces") or [])]
        faces = [x for x in faces if area(x[0]) > 0]
        dets, used = [], set()
        for (box, score, sharp), tid in zip(faces, faces_l.update(f, [x[0] for x in faces])):
            inside = [k for k, d in enumerate(bodies) if k not in used and _inside(_centre(box), d["bbox"])]
            body = min(inside, key=lambda k: area(bodies[k]["bbox"])) if inside else None
            if body is not None:
                used.add(body)
            dets.append({"track_id": tid, "cname": "person",
                         "bbox": list(bodies[body]["bbox"]) if body is not None else list(box),
                         "conf": float(bodies[body]["conf"]) if body is not None else float(score),
                         "face_bbox": list(box), "face_conf": float(score), "sharp": float(sharp)})
        rest = [d for k, d in enumerate(bodies) if k not in used]
        for d, tid in zip(rest, bodies_l.update(f, [d["bbox"] for d in rest])):
            dets.append({"track_id": tid, "cname": "body", "bbox": list(d["bbox"]), "conf": float(d["conf"])})
        for d, tid in zip(objects, objects_l.update(f, [d["bbox"] for d in objects])):
            dets.append({"track_id": tid, "cname": d["cname"], "bbox": list(d["bbox"]), "conf": float(d["conf"])})
        out[f] = dets
    return out
