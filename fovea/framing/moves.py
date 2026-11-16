"""How the camera moves inside ONE shot: hold, pan, or cut. Plain Python (no numpy / pandas).

Input: the focal x position the camera would like per frame (it can be jumpy: detector jitter, the
subject flipping between two candidates). Output: where the camera actually is per frame.

Rules, in order:
  * a short shot (under ``short_shot`` frames) gets ONE framing for its whole length: there is no
    time for a move to read as anything but a jolt. It frames a person if one is there for a real
    part of the shot, otherwise whatever was followed longest
  * the camera holds while the subject stays within a dead band around it
  * it only moves once the subject has stayed away for ``settle`` frames in a row (a flicker of
    another subject is ignored), and then goes to where the subject is over those frames
  * far and onto a speaker -> cut; otherwise an eased pan whose length grows with the distance,
    so a long move is never a whip
"""

import math


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2.0


def plan_shot(raw, tags, min_cx, max_cx, crop_w, pan_window=12, cut_threshold=450.0, deadband=45.0,
              spike=200.0, settle=10, short_shot=38):
    """``raw``: wanted x per frame; ``tags``: what is being followed per frame (``speaker_3``, ...).
    Returns ``(cx, modes)``, one entry per frame."""
    n = len(raw)
    if n == 0:
        return [], []

    def clip(x):
        return min(max(x, min_cx), max_cx)

    # isolated 1-2 frame spikes -> the local median
    tgt = []
    for i in range(n):
        med = _median(raw[max(0, i - 2):min(n, i + 3)])
        tgt.append(clip(med if abs(raw[i] - med) > spike else raw[i]))

    band = max(deadband, 0.12 * crop_w)      # the subject is still well inside the crop at 12% off-centre

    if n < short_shot:                       # one framing: where the subject followed longest is...
        counts = {}
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
        # ...but a person who is there for a real part of the shot comes before text, motion or the
        # frame centre, even if those were followed for more frames (a person seen late in the shot)
        enough = max(5, int(0.2 * n))
        people = {t: c for t, c in counts.items() if c >= enough and str(t).startswith(("speaker", "person", "body"))}
        pool = people or counts
        top = max(pool, key=lambda t: (str(t).startswith("speaker"), pool[t], -list(tags).index(t)))
        pos = _median([x for x, t in zip(tgt, tags) if t == top])
        return [pos] * n, ["shot_start_cut"] + ["camera_hold"] * (n - 1)

    cx, modes = [0.0] * n, [""] * n
    cx[0] = _median(tgt[:settle])            # start where the subject settles, not on a first-frame blip
    modes[0] = "shot_start_cut"
    i = 1
    while i < n:
        prev = cx[i - 1]
        ahead = tgt[i:i + settle]
        if abs(tgt[i] - prev) <= band or len(ahead) < settle or not all(abs(x - prev) > band for x in ahead):
            cx[i], modes[i] = prev, "camera_hold"
            i += 1
            continue
        dest = _median(ahead)
        dist = abs(dest - prev)
        if dist > cut_threshold and tags[i] != tags[i - 1] and "speaker" in str(tags[i]):
            cx[i], modes[i] = dest, "instant_cut"
            i += 1
            continue
        window = min(n - i, max(2, int(round(pan_window * max(1.0, dist / (0.5 * crop_w))))))
        for k in range(1, window + 1):
            # cosine ease in / out
            ease = 0.5 * (1.0 - math.cos(math.pi * k / window))
            cx[i + k - 1] = prev + (dest - prev) * ease
            modes[i + k - 1] = "smooth_pan"
        i += window
    return [clip(x) for x in cx], modes

