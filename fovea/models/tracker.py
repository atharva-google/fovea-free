"""Person tracking with face boxes, for many shots at once.

How a shard is tracked (``PersonTracker.track_shots``):
  * several decoder threads each read a different run of shots from the proxy and push the
    frames the detector needs (every ``detect_every``-th frame of a shot) into one queue
  * the main thread takes frames from that queue in **batches that may mix shots** and runs ONE
    YOLO forward pass per batch (FP16 on CUDA). Detection is stateless, so mixing shots is safe.
  * every detection is routed to the ByteTrack instance of ITS shot. One decoder reads a shot
    from start to end, so a shot's frames always arrive in order (ByteTrack needs that).
  * MediaPipe face detection (CPU) on each person crop runs in a thread pool, overlapping with
    the following YOLO batches. ``face_backend="pose"`` instead takes the face from the head
    keypoints of a YOLO-pose model, in the same GPU pass.
  * IN ADDITION, MediaPipe looks for faces on the WHOLE frame (short-range model for close-ups +
    the full-range one). Such a face is used ONLY where the steps above found no face: a close-up
    with no body for YOLO to find, or a person whose face was missed inside their box. Everything
    the old path finds is kept exactly as it was, so speaker detection sees the same face tracks.
  * frames the detector skipped are filled by interpolating each track (``fill_skipped_frames``)

Output structure (``ShotTracks``)::

    {shot_id: {frame_idx: [ detections ]}}

    {track_id, cname: "person", bbox, conf, face_bbox, face_conf, sharp}   a person with a face
    {track_id, cname: "body", bbox, conf}                                 a person with no face found
    {track_id, cname: <yolo class>, bbox, conf}                           an object

Faces added from the whole frame have track ids from 5000 (linked by ``faces.Linker``).
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
from .faces import add_missed_faces

ShotTracks = Dict[int, Dict[int, List[Dict[str, Any]]]]
FrameTracks = Dict[int, List[Dict[str, Any]]]

_DONE = object()


def _make_bytetrack(frame_rate: int):
    """Build Ultralytics' ByteTrack directly so we can feed it batched detections."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    path = check_yaml("bytetrack.yaml")
    try:  # newer ultralytics
        from ultralytics.utils import YAML

        cfg = YAML.load(path)
    except ImportError:  # older ultralytics
        from ultralytics.utils import yaml_load

        cfg = yaml_load(path)

    import inspect

    if "frame_rate" in inspect.signature(BYTETracker.__init__).parameters:  # older ultralytics
        return BYTETracker(args=IterableSimpleNamespace(**cfg), frame_rate=frame_rate)
    # Newer ultralytics dropped ``frame_rate`` and uses ``track_buffer`` as-is. Older versions kept
    # lost tracks for ``track_buffer * frame_rate / 30`` frames, so apply that scaling here to keep
    # the same behaviour (we track at fps / detect_every, not 30 fps).
    cfg["track_buffer"] = max(1, int(frame_rate / 30.0 * cfg.get("track_buffer", 30)))
    return BYTETracker(args=IterableSimpleNamespace(**cfg))


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
        face_margin: float = 0.0,
    ):
        from ultralytics import YOLO

        _quiet_half_warning()
        self.model = YOLO(weights_path)  # detection model, or pose model for face_backend="pose"
        self.device = device
        self.half = device == "cuda"
        self.conf, self.iou = conf, iou
        self.face_backend = face_backend
        self.face_model_path = face_model_path
        # whole-frame face search: short-range first (close-ups), then full-range
        self.full_frame_models = [str(p) for p in (face_short_model_path, face_model_path) if p]
        self.face_margin = face_margin
        self.face_min_conf = face_min_conf
        self.detect_every = max(1, int(detect_every))
        self.frame_rate = max(1, int(round(fps / self.detect_every)))
        self._face_local = threading.local()  # one MediaPipe detector per face thread
        self._lock = threading.Lock()
        self.seconds = {"decode_wait": 0.0, "yolo": 0.0, "bytetrack": 0.0, "faces_cpu": 0.0, "faces_wait": 0.0}

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
        bytetracks: Dict[int, Any] = {}
        max_pending_faces = min(max_pending_faces, 96)  # whole-frame face jobs each hold a frame
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
                    self._process_batch(batch, bytetracks, raw, face_pool, pending, max_pending_faces)
                    if on_frames:
                        on_frames(len(batch))
                    batch = []
            if batch:
                self._process_batch(batch, bytetracks, raw, face_pool, pending, max_pending_faces)
                if on_frames:
                    on_frames(len(batch))
            t0 = time.perf_counter()
            result = {}
            for sid, frames in raw.items():
                resolved = {idx: {"dets": _resolve(e["dets"]), "full": e["full"].result() if e["full"] is not None else []}
                            for idx, e in frames.items()}
                result[sid] = add_missed_faces(resolved)
            self.seconds["faces_wait"] += time.perf_counter() - t0
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
    def _process_batch(self, batch, bytetracks, raw, face_pool, pending: deque, max_pending: int) -> None:
        t0 = time.perf_counter()
        extra = {"half": True} if self.half else {}  # passing half=False only triggers a deprecation warning
        results = self.model.predict(
            source=[img for _, _, img in batch], conf=self.conf, iou=self.iou,
            device=self.device, verbose=False, **extra,
        )
        t1 = time.perf_counter()
        self.seconds["yolo"] += t1 - t0

        for (sid, frame_idx, img), res in zip(batch, results):
            det = res.boxes.cpu().numpy()
            current: List[Any] = []
            if len(det):
                tracker = bytetracks.get(sid)
                if tracker is None:  # fresh tracker per shot so IDs don't leak across cuts
                    tracker = bytetracks[sid] = _make_bytetrack(self.frame_rate)
                tracks = tracker.update(det, img)
                kpts = res.keypoints.data.cpu().numpy() if getattr(res, "keypoints", None) is not None else None
                h, w = img.shape[:2]
                for t in tracks:
                    # tracks rows: x1, y1, x2, y2, track_id, score, cls, det_index
                    box = [int(v) for v in t[:4]]
                    tid, score, cls_id, det_idx = int(t[4]), float(t[5]), int(t[6]), int(t[7])
                    cname = res.names[cls_id]
                    if cname != "person":
                        current.append({"track_id": tid, "bbox": box, "conf": score, "cname": cname})
                        continue
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    base = {"track_id": tid, "bbox": [x1, y1, x2, y2], "conf": score, "cname": "person"}
                    if self.face_backend == "pose":
                        face = _face_from_keypoints(kpts[det_idx], base["bbox"], w, h) if kpts is not None else None
                        if face is not None:
                            fx1, fy1, fx2, fy2 = face[0]
                            current.append({**base, "face_bbox": face[0], "face_conf": face[1],
                                            "sharp": _sharpness(img[fy1:fy2, fx1:fx2])})
                        else:
                            current.append({**base, "cname": "body"})
                    else:
                        crop = np.ascontiguousarray(img[y1:y2, x1:x2])  # copy: don't keep the frame alive
                        fut = face_pool.submit(self._mediapipe_face, crop, x1, y1)
                        current.append((fut, base))
                        pending.append(fut)
            full = None
            if face_pool is not None and self.full_frame_models:
                full = face_pool.submit(self._full_frame_faces, img)
                pending.append(full)
            raw[sid][frame_idx] = {"dets": current, "full": full}

        # keep the number of in-flight face crops bounded (memory)
        while len(pending) > max_pending:
            f0 = time.perf_counter()
            pending.popleft().result()
            self.seconds["faces_wait"] += time.perf_counter() - f0
        while pending and pending[0].done():
            pending.popleft()
        self.seconds["bytetrack"] += time.perf_counter() - t1

    def _full_frame_faces(self, img: np.ndarray) -> List[Tuple[List[int], float, float]]:
        """Every face on the whole frame, from each model: ``[(box, score, sharpness)]``."""
        t0 = time.perf_counter()
        try:
            import mediapipe as mp

            detectors = getattr(self._face_local, "full_detectors", None)
            if detectors is None:
                from mediapipe.tasks.python import BaseOptions, vision

                detectors = self._face_local.full_detectors = [
                    vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                        base_options=BaseOptions(model_asset_path=path),
                        running_mode=vision.RunningMode.IMAGE,
                        min_detection_confidence=self.face_min_conf,
                    )) for path in self.full_frame_models]
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

    def _mediapipe_face(self, crop: np.ndarray, ox: int, oy: int) -> Optional[Tuple[List[int], float, float]]:
        t0 = time.perf_counter()
        try:
            detector = getattr(self._face_local, "detector", None)
            if detector is None:
                from mediapipe.tasks.python import BaseOptions, vision

                detector = self._face_local.detector = vision.FaceDetector.create_from_options(
                    vision.FaceDetectorOptions(
                        base_options=BaseOptions(model_asset_path=str(self.face_model_path)),
                        running_mode=vision.RunningMode.IMAGE,
                        min_detection_confidence=self.face_min_conf,
                    )
                )
            return _face_from_mediapipe_crop(crop, ox, oy, detector)
        finally:
            with self._lock:
                self.seconds["faces_cpu"] += time.perf_counter() - t0


def _quiet_half_warning() -> None:
    """Newer Ultralytics logs "'half' is deprecated" on EVERY predict call; keep the logs readable."""
    import logging

    class _Drop(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "'half' is deprecated" not in record.getMessage()

    log = logging.getLogger("ultralytics")
    if not any(isinstance(f, _Drop) or type(f).__name__ == "_Drop" for f in log.filters):
        log.addFilter(_Drop())


def _sharpness(crop: np.ndarray) -> float:
    """How in-focus a face is: edge energy of the crop at a fixed small size (so big and small faces
    compare). A blurred background face scores far below the in-focus subject."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _resolve(entries: List[Any]) -> List[Dict[str, Any]]:
    """Turn (face_future, person_dict) placeholders into final detections (in original order).
    A person whose face was not found is kept as a ``body`` (it used to be dropped)."""
    out = []
    for e in entries:
        if isinstance(e, tuple):
            fut, base = e
            face = fut.result()
            if face is not None:
                out.append({**base, "face_bbox": face[0], "face_conf": face[1], "sharp": face[2]})
            else:
                out.append({**base, "cname": "body"})
        else:
            out.append(e)
    return out


def _face_from_mediapipe_crop(crop: np.ndarray, px1: int, py1: int,
                              detector) -> Optional[Tuple[List[int], float, float]]:
    import mediapipe as mp

    h, w = crop.shape[:2]
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
    if not res.detections:
        return None
    # Face closest to where a head usually is (top-center of the body box)
    tx, ty = w * 0.5, h * 0.2
    best = min(res.detections, key=lambda d: (
        (d.bounding_box.origin_x + d.bounding_box.width * 0.5 - tx) ** 2
        + (d.bounding_box.origin_y + d.bounding_box.height * 0.5 - ty) ** 2))
    bb = best.bounding_box
    fx1, fy1 = max(0, bb.origin_x), max(0, bb.origin_y)
    fx2, fy2 = min(w, fx1 + bb.width), min(h, fy1 + bb.height)
    return ([px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2], float(best.categories[0].score),
            _sharpness(crop[fy1:fy2, fx1:fx2]))


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
