"""Faces, people and objects per shot, for many shots at once.

How a shard is analysed (``PersonTracker.track_shots``):
  * several decoder threads each read a different run of shots from the proxy and push the
    frames the detectors need (every ``detect_every``-th frame of a shot) into one queue
  * the main thread takes frames from that queue in **batches that may mix shots** and runs ONE
    YOLO forward pass per batch (FP16 on CUDA): people (kept as a fallback) and objects
  * FACES are found on the WHOLE frame by MediaPipe (CPU, in a thread pool, overlapping with the
    following YOLO batches): the short-range model for close-ups and the full-range model for the
    rest. A face no longer needs a YOLO person box around it, so a close-up with no body is found.
    ``face_backend="pose"`` instead takes faces from the head keypoints of a YOLO-pose model.
  * per shot, faces are merged, given a margin and linked into tracks by our own linker
    (``faces.py``; there is no ByteTrack any more)
  * frames the detectors skipped are filled by interpolating each track (``fill_skipped_frames``)

Output structure (``ShotTracks``)::

    {shot_id: {frame_idx: [ detections ]}}

    {track_id, cname: "person", bbox, conf, face_bbox, face_conf, sharp}   a face (bbox = its body, if YOLO saw one)
    {track_id, cname: "body", bbox, conf}                                 a person with no visible face
    {track_id, cname: <yolo class>, bbox, conf}                           an object
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.video import ShotReader, split_contiguous
from .faces import assemble_shot

ShotTracks = Dict[int, Dict[int, List[Dict[str, Any]]]]
FrameTracks = Dict[int, List[Dict[str, Any]]]

_DONE = object()


class PersonTracker:
    def __init__(
        self,
        weights_path: str,
        device: str,
        fps: float,
        face_backend: str = "mediapipe",
        face_model_path: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.5,
        face_min_conf: float = 0.5,
        detect_every: int = 1,
        face_short_model_path: Optional[str] = None,
        face_margin: float = 0.15,
    ):
        from ultralytics import YOLO

        _quiet_half_warning()
        self.model = YOLO(weights_path)  # detection model, or pose model for face_backend="pose"
        self.device = device
        self.half = device == "cuda"
        self.conf, self.iou = conf, iou
        self.face_backend = face_backend
        self.face_model_path = face_model_path
        # short-range first (close-ups), then full-range; the same face found by both is merged later
        self.face_model_paths = [str(p) for p in (face_short_model_path, face_model_path) if p]
        self.face_margin = face_margin
        self.frame_size: Optional[Tuple[int, int]] = None
        self.face_min_conf = face_min_conf
        self.detect_every = max(1, int(detect_every))
        self.frame_rate = max(1, int(round(fps / self.detect_every)))
        self._face_local = threading.local()  # one MediaPipe detector per face thread
        self._lock = threading.Lock()
        self.seconds = {"decode_wait": 0.0, "yolo": 0.0, "link": 0.0, "faces_cpu": 0.0, "faces_wait": 0.0}

    # ------------------------------------------------------------- public API
    def track_shots(self, video_path: str, shots: Sequence[Dict[str, Any]], batch_size: int = 16,
                    decoders: int = 2, face_workers: int = 2, max_pending_faces: int = 512,
                    on_frames: Optional[Callable[[int], None]] = None) -> ShotTracks:
        """Detect + track every shot in ``shots`` (dicts with shot_id/start_frame/end_frame).

        Returns RAW tracks (only detected frames); see ``fill_skipped_frames``.
        ``on_frames(n)`` is called after every YOLO batch with the number of frames it held.
        """
        shots = sorted(shots, key=lambda s: int(s["start_frame"]))
        raw: Dict[int, Dict[int, list]] = {int(s["shot_id"]): {} for s in shots}
        if not shots:
            return raw

        # ---- decoder threads: runs of consecutive shots, handed out from a queue
        work: "queue.Queue" = queue.Queue()
        for group in split_contiguous(shots, max(1, decoders) * 4):
            work.put(group)
        n_dec = max(1, min(decoders, work.qsize()))
        frames_q: "queue.Queue" = queue.Queue(maxsize=max(8, batch_size * 3))
        stop = threading.Event()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    frames_q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def decoder() -> None:
            reader = None
            try:
                reader = ShotReader(video_path)
                every = self.detect_every
                while not stop.is_set():
                    try:
                        group = work.get_nowait()
                    except queue.Empty:
                        break
                    for shot in group:
                        sid, s0, s1 = int(shot["shot_id"]), int(shot["start_frame"]), int(shot["end_frame"])
                        for idx, frame in reader.frames(s0, s1, want=lambda i, s0=s0: (i - s0) % every == 0):
                            if not put((sid, idx, frame)):
                                return
            except BaseException as e:  # noqa: BLE001 - handed to the main thread
                put(e)
            finally:
                if reader is not None:
                    reader.close()
                put(_DONE)

        threads = [threading.Thread(target=decoder, daemon=True, name=f"yolo-decode-{i}") for i in range(n_dec)]
        for t in threads:
            t.start()

        face_pool = (ThreadPoolExecutor(max_workers=max(1, face_workers), thread_name_prefix="faces")
                     if self.face_backend == "mediapipe" else None)
        max_pending_faces = min(max_pending_faces, 64)  # each pending job holds a whole frame
        pending: deque = deque()  # face futures not yet known to be finished (bounded)
        try:
            batch: List[Tuple[int, int, np.ndarray]] = []
            finished = 0
            while finished < n_dec:
                t0 = time.perf_counter()
                item = frames_q.get()
                self.seconds["decode_wait"] += time.perf_counter() - t0
                if item is _DONE:
                    finished += 1
                    continue
                if isinstance(item, BaseException):
                    raise item
                batch.append(item)
                if len(batch) >= batch_size:
                    self._process_batch(batch, raw, face_pool, pending, max_pending_faces)
                    if on_frames:
                        on_frames(len(batch))
                    batch = []
            if batch:
                self._process_batch(batch, raw, face_pool, pending, max_pending_faces)
                if on_frames:
                    on_frames(len(batch))
            t0 = time.perf_counter()
            resolved = {sid: {idx: _resolve(entry) for idx, entry in frames.items()} for sid, frames in raw.items()}
            self.seconds["faces_wait"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            w, h = self.frame_size or (0, 0)
            result = {sid: assemble_shot(frames, w, h, self.face_margin) for sid, frames in resolved.items()}
            self.seconds["link"] += time.perf_counter() - t0
            return result
        finally:
            stop.set()
            while any(t.is_alive() for t in threads):  # unblock decoders stuck on a full queue
                try:
                    frames_q.get(timeout=0.05)
                except queue.Empty:
                    pass
            if face_pool is not None:
                face_pool.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------- internals
    def _process_batch(self, batch, raw, face_pool, pending: deque, max_pending: int) -> None:
        t0 = time.perf_counter()
        extra = {"half": True} if self.half else {}  # passing half=False only triggers a deprecation warning
        results = self.model.predict(
            source=[img for _, _, img in batch], conf=self.conf, iou=self.iou,
            device=self.device, verbose=False, **extra,
        )
        self.seconds["yolo"] += time.perf_counter() - t0

        for (sid, frame_idx, img), res in zip(batch, results):
            h, w = img.shape[:2]
            self.frame_size = (w, h)
            det = res.boxes.cpu().numpy()
            kpts = res.keypoints.data.cpu().numpy() if getattr(res, "keypoints", None) is not None else None
            yolo: List[Dict[str, Any]] = []
            pose_faces: List[Tuple[List[int], float, float]] = []
            for i in range(len(det)):
                x1, y1, x2, y2 = (int(v) for v in det.xyxy[i][:4])
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                cname = res.names[int(det.cls[i])]
                yolo.append({"bbox": [x1, y1, x2, y2], "conf": float(det.conf[i]),
                             "cname": "body" if cname == "person" else cname})
                if cname == "person" and self.face_backend == "pose" and kpts is not None:
                    face = _face_from_keypoints(kpts[i], [x1, y1, x2, y2], w, h)
                    if face is not None:
                        fx1, fy1, fx2, fy2 = face[0]
                        pose_faces.append((face[0], face[1], _sharpness(img[fy1:fy2, fx1:fx2])))
            if face_pool is not None:
                faces: Any = face_pool.submit(self._mediapipe_faces, img)   # the WHOLE frame, not person crops
                pending.append(faces)
            else:
                faces = pose_faces
            raw[sid][frame_idx] = {"faces": faces, "yolo": yolo}

        # keep the number of in-flight face jobs bounded (each holds a frame)
        while len(pending) > max_pending:
            f0 = time.perf_counter()
            pending.popleft().result()
            self.seconds["faces_wait"] += time.perf_counter() - f0
        while pending and pending[0].done():
            pending.popleft()

    def _mediapipe_faces(self, img: np.ndarray) -> List[Tuple[List[int], float, float]]:
        """Every face on the frame, from each face model: ``[(box, score, sharpness)]``."""
        t0 = time.perf_counter()
        try:
            import mediapipe as mp

            detectors = getattr(self._face_local, "detectors", None)
            if detectors is None:
                from mediapipe.tasks.python import BaseOptions, vision

                detectors = self._face_local.detectors = [
                    vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                        base_options=BaseOptions(model_asset_path=path),
                        running_mode=vision.RunningMode.IMAGE,
                        min_detection_confidence=self.face_min_conf,
                    )) for path in self.face_model_paths]
            h, w = img.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            found: List[Tuple[List[int], float, float]] = []
            for detector in detectors:
                for d in detector.detect(image).detections:
                    bb = d.bounding_box
                    x1, y1 = max(0, int(bb.origin_x)), max(0, int(bb.origin_y))
                    x2, y2 = min(w, int(bb.origin_x + bb.width)), min(h, int(bb.origin_y + bb.height))
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    found.append(([x1, y1, x2, y2], float(d.categories[0].score), _sharpness(img[y1:y2, x1:x2])))
            return found
        finally:
            with self._lock:
                self.seconds["faces_cpu"] += time.perf_counter() - t0


def _sharpness(crop: np.ndarray) -> float:
    """How in-focus a face is: edge energy of the crop at a fixed small size (so big and small faces
    compare). A blurred background face scores far below the in-focus subject."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _quiet_half_warning() -> None:
    """Newer Ultralytics logs "'half' is deprecated" on EVERY predict call; keep the logs readable."""
    import logging

    class _Drop(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "'half' is deprecated" not in record.getMessage()

    log = logging.getLogger("ultralytics")
    if not any(isinstance(f, _Drop) or type(f).__name__ == "_Drop" for f in log.filters):
        log.addFilter(_Drop())


def _resolve(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Wait for a frame's face job (if it is one) and return ``{"faces": [...], "yolo": [...]}``."""
    faces = entry["faces"]
    return {"faces": faces.result() if hasattr(faces, "result") else faces, "yolo": entry["yolo"]}


def _face_from_keypoints(kpts: np.ndarray, box, frame_w: int, frame_h: int,
                         min_conf: float = 0.5) -> Optional[Tuple[List[int], float]]:
    """Approximate a face box from COCO head keypoints (nose, eyes, ears)."""
    head = kpts[:5]
    vis = head[head[:, 2] >= min_conf]
    if len(vis) < 2:
        return None
    cx, cy = float(vis[:, 0].mean()), float(vis[:, 1].mean())
    span = float(vis[:, 0].max() - vis[:, 0].min())
    shoulders = kpts[5:7]
    shoulder_w = float(abs(shoulders[0, 0] - shoulders[1, 0])) if (shoulders[:, 2] >= min_conf).all() else 0.0
    size = max(span * 1.8, shoulder_w * 0.5, (box[2] - box[0]) * 0.2, 8.0)
    x1, x2 = cx - size / 2, cx + size / 2
    y1, y2 = cy - size * 0.55, cy + size * 0.65
    return ([int(max(0, x1)), int(max(0, y1)), int(min(frame_w, x2)), int(min(frame_h, y2))],
            float(vis[:, 2].mean()))


# ------------------------------------------------------------------ frame-skip fill
def _lerp_box(a, b, t):
    return [int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4)]


def fill_skipped_frames(frame_tracks: FrameTracks, start: int, end: int) -> FrameTracks:
    """Fill frames that the detector skipped by interpolating each track between
    its surrounding detected frames (or holding the last value at the shot end)."""
    detected = sorted(frame_tracks)
    if not detected:
        return {f: [] for f in range(start, end)}

    filled: FrameTracks = dict(frame_tracks)
    for f in range(start, detected[0]):
        filled[f] = []
    for d0, d1 in zip(detected, detected[1:] + [None]):
        stop = d1 if d1 is not None else end
        if stop - d0 <= 1:
            continue
        nxt = {d["track_id"]: d for d in frame_tracks[d1]} if d1 is not None else {}
        for f in range(d0 + 1, stop):
            t = (f - d0) / (d1 - d0) if d1 is not None else 0.0
            dets = []
            for a in frame_tracks[d0]:
                b = nxt.get(a["track_id"])
                if b is None:
                    dets.append(dict(a))  # hold
                    continue
                d = dict(a)
                d["bbox"] = _lerp_box(a["bbox"], b["bbox"], t)
                if a.get("face_bbox") is not None and b.get("face_bbox") is not None:
                    d["face_bbox"] = _lerp_box(a["face_bbox"], b["face_bbox"], t)
                dets.append(d)
            filled[f] = dets
    return filled
