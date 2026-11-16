"""Who is the subject: stricter "is speaking" labels, and how prominent a silent person is.

Plain Python (no numpy / pandas), tested in tests/test_subjects.py.

Speaker labels come from TalkNet, which scores every face track on every frame; a raw score above 0
used to mean "speaking". With several faces in a shot that flickers: one stray positive frame on a
background face was enough to pull the camera. Here a face counts as speaking only when

  * its score, averaged over about a third of a second, is positive
  * it is the strongest such face on that frame (one speaker at a time)
  * that holds for a stretch (short gaps are bridged, short bursts are dropped)
  * the face is not small next to the largest face on screen (people behind the subject)

The rest of this file is the per-shot preparation the framing does before it picks a path:
holding people through frames where the detector lost them, ignoring a logo that sits in the same
corner in every shot, and finding where the movement of a whole shot is.
"""


def speaking_frames(scored_tracks, face_area=None, window=9, min_run=8, max_hole=8, rel_face=0.5,
                    min_score=0.1):
    """``scored_tracks``: ``[{"shot_id", "track_id", "frames": [...], "scores": [...]}]`` (TalkNet).
    ``face_area``: optional ``{(shot_id, frame, track_id): area}`` of each face box.
    Returns ``{(shot_id, frame, track_id): score}`` for the frames in which that face is speaking."""
    face_area = face_area or {}
    biggest = {}
    for (shot, frame, _), area in face_area.items():
        if area > biggest.get((shot, frame), 0.0):
            biggest[(shot, frame)] = area

    smoothed = []                                   # per track: (shot, track, frames, averaged scores)
    half = window // 2
    for trk in scored_tracks:
        shot, tid = int(trk["shot_id"]), trk["track_id"]
        frames = [int(f) for f in trk["frames"]]
        raw = [float(s) for s in trk["scores"]][:len(frames)]
        frames = frames[:len(raw)]
        avg = []
        for i in range(len(raw)):
            part = raw[max(0, i - half):i + half + 1]
            value = sum(part) / len(part)
            area = face_area.get((shot, frames[i], tid))
            if area is not None and area < rel_face * biggest.get((shot, frames[i]), 0.0):
                value = 0.0                         # someone behind the subject
            if value < min_score:
                value = 0.0                         # barely positive is not speech
            avg.append(value)
        smoothed.append((shot, tid, frames, avg))

    best = {}                                       # one speaker at a time: the strongest face per frame
    for shot, tid, frames, avg in smoothed:
        for f, v in zip(frames, avg):
            if v > 0 and v > best.get((shot, f), (0.0, None))[0]:
                best[(shot, f)] = (v, tid)

    out = {}
    for shot, tid, frames, avg in smoothed:
        on = [v > 0 and best.get((shot, f), (0.0, None))[1] == tid for f, v in zip(frames, avg)]
        i, n = 0, len(on)
        while i < n:                                # bridge short gaps inside a stretch of speech
            if not on[i] and i > 0 and on[i - 1]:
                j = i
                while j < n and not on[j]:
                    j += 1
                if j < n and j - i <= max_hole:
                    for k in range(i, j):
                        on[k] = True
                i = j
            else:
                i += 1
        i = 0
        while i < n:                                # keep only stretches long enough to be speech
            if on[i]:
                j = i
                while j < n and on[j]:
                    j += 1
                if j - i >= min_run:
                    for k in range(i, j):
                        out[(shot, frames[k], tid)] = round(max(avg[k], 0.001), 3)
                i = j
            else:
                i += 1
    return out


def speaker_score(prominence, rel_face=1.0, asd=0.0):
    """Speaking is a boost on top of how prominent the person is, not a trump card.

    ``rel_face``: their face area next to the largest face on screen (1 = the largest). Two people
    of similar size in a conversation: the one speaking clearly wins (about double). A small face
    behind a large one gets little from speaking and does not pull the camera off the subject."""
    rel = min(1.0, max(0.0, rel_face))
    return prominence * (1.0 + rel * rel * (1.2 + 0.2 * min(1.0, max(0.0, asd))))


def fill_track_gaps(frames_data, shot_frames, max_gap=25):
    """A person the detector loses for a few frames is still there.

    Returns a copy of ``frames_data`` ({frame: [detections]}) for this shot in which
      * a track missing between two of its appearances (up to ``max_gap`` frames apart) is filled
        in, its boxes moving evenly from one to the other
      * a frame with nobody at all borrows the people of the nearest frame that has some (up to
        ``max_gap`` frames away), so a dropout is not a chance for text or an object to take over
    Filled detections are marked ``"filled": True`` and never count as speaking."""
    frames = [int(f) for f in shot_frames]
    out = {f: list(frames_data.get(f, [])) for f in frames}

    def lerp(a, b, w):
        return [a[k] + (b[k] - a[k]) * w for k in range(4)]

    seen = {}
    for f in frames:
        for d in out[f]:
            seen.setdefault(d["track_id"], []).append((f, d))
    for tid, items in seen.items():
        for (fa, da), (fb, db) in zip(items, items[1:]):
            if not 1 < fb - fa <= max_gap:
                continue
            for f in range(fa + 1, fb):
                if f not in out:
                    continue
                w = (f - fa) / float(fb - fa)
                d = {**da, "bbox": lerp(da["bbox"], db["bbox"], w), "filled": True, "speaking": False, "asd_score": None}
                if da.get("face_bbox") is not None and db.get("face_bbox") is not None:
                    d["face_bbox"] = lerp(da["face_bbox"], db["face_bbox"], w)
                out[f].append(d)

    def people(f):
        return [d for d in out[f] if d.get("cname") == "person"]

    with_people = [f for f in frames if people(f)]
    for f in frames:
        if people(f) or not with_people:
            continue
        near = min(with_people, key=lambda g: (abs(g - f), g))
        if abs(near - f) <= max_gap:
            out[f] = out[f] + [{**d, "filled": True, "speaking": False, "asd_score": None} for d in people(near)]
    return out


def watermark_indices(boxes, n_shots=None, tol=0.04, min_shots=3, min_share=0.0):
    """Text that sits in the same place in many shots is a logo / watermark, not something to look at.

    ``boxes``: ``[(shot_id, cx, cy)]`` with the centre of each text box as a share of the frame (0-1).
    Returns the indices of the boxes to ignore."""
    shots = {s for s, _, _ in boxes}
    need = max(min_shots, int(min_share * max(n_shots or 0, len(shots)) + 0.999))
    drop = set()
    for i, (_, x, y) in enumerate(boxes):
        same_place = {s for s, x2, y2 in boxes if abs(x2 - x) <= tol and abs(y2 - y) <= tol}
        if len(same_place) >= need:
            drop.add(i)
    return drop


def shot_motion_cx(motion_results, shot_frames, min_density=0.05, min_share=0.2):
    """Where the movement of a whole shot is: slow movement that never crosses the per-frame
    threshold still adds up. Returns an x position, or None when the shot hardly moves."""
    total = weight = 0.0
    moving = 0
    for f in shot_frames:
        m = motion_results.get(f)
        if not m or m.get("motion_cx") is None:
            continue
        d = float(m.get("instant_density", 0.0))
        if d >= min_density:
            moving += 1
            total += d * float(m["motion_cx"])
            weight += d
    if not weight or moving < min_share * max(1, len(shot_frames)):
        return None
    return total / weight


def person_score(area_ratio, center_offset, presence=1.0, rel_sharp=1.0, full=0.25):
    """How strongly a person who is NOT speaking attracts the camera (up to 0.9; speaking multiplies it).

    ``area_ratio``: their FACE box as a share of the frame (``full`` = the share at which bigger
    stops mattering); ``center_offset``: 0 in the middle of the frame, 1 at an edge; ``presence``:
    share of the shot they are on screen for; ``rel_sharp``: how in-focus their face is next to the
    sharpest face on screen (1 = the sharpest). A large, central, in-focus face that is there for
    the whole shot comes first; a small, blurred face at the edge comes last."""
    size = min(1.0, max(0.0, area_ratio) / full)
    central = 1.0 - min(1.0, max(0.0, center_offset))
    base = 0.40 + 0.30 * size + 0.12 * central + 0.08 * min(1.0, max(0.0, presence))
    return base * (0.55 + 0.45 * min(1.0, max(0.0, rel_sharp)))
