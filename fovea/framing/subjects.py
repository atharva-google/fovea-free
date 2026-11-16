"""Who is the subject: stricter "is speaking" labels, and how prominent a silent person is.

Plain Python (no numpy / pandas), tested in tests/test_subjects.py.

Speaker labels come from TalkNet, which scores every face track on every frame; a raw score above 0
used to mean "speaking". With several faces in a shot that flickers: one stray positive frame on a
background face was enough to pull the camera. Here a face counts as speaking only when

  * its score, averaged over about a third of a second, is positive
  * it is the strongest such face on that frame (one speaker at a time)
  * that holds for a stretch (short gaps are bridged, short bursts are dropped)
  * the face is not tiny next to the largest face on screen (background people)
"""


def speaking_frames(scored_tracks, face_area=None, window=9, min_run=8, max_hole=8, rel_face=0.25):
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
                value = 0.0                         # a background face
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


def person_score(area_ratio, center_offset, presence=1.0):
    """How strongly a person who is NOT speaking attracts the camera (0.5 - 0.9; a speaker scores 2+).

    ``area_ratio``: their box as a share of the frame; ``center_offset``: 0 in the middle of the
    frame, 1 at an edge; ``presence``: share of the shot they are on screen for. Large, central
    people who are there for the whole shot come first; someone passing at the edge comes last."""
    size = min(1.0, max(0.0, area_ratio) / 0.25)
    central = 1.0 - min(1.0, max(0.0, center_offset))
    return 0.50 + 0.22 * size + 0.10 * central + 0.08 * min(1.0, max(0.0, presence))
