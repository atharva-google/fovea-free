"""The output schemas of the two Gemini calls. Gemini is made to answer in exactly these shapes.

    call 1  Units      once per input FILE (stored): describes every unit of that file
    call 2  SyncedEdits / CutawayEdits   text only: one edit per requested length (one request per mode)

Call 1 is the only time anything LOOKS at the footage: it says what happens in every unit and
which earlier units it depends on. Call 2 decides what to keep.

Units are cut by CODE before call 1, on two tracks that each cover the whole file:

    video units  V01, V02, ...  one per shot
    audio units  A01, A02, ...  stretches of sound: a spoken phrase, or the sound between phrases
                                (a phrase running across a shot cut stays ONE audio unit)

When stored, ids get the file in front (``F1.V03``), so call 2 can mix files. Gemini never writes
a timestamp in either call: it only names units. Every time comes from the code.

What CODE adds to every stored unit (never asked of Gemini): start, end, length, shot, which video
units an audio unit was recorded over, whether a face is visibly talking, the detected focus points,
and how the unit crops in each requested aspect ratio.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


# =========================================================================== call 1: Units (per file)
class Music(str, Enum):
    none = "none"  # no continuous music
    bed = "bed"    # continuous music runs under the file
    only = "only"  # the file IS music (a supplied music track)


class VideoUnit(BaseModel):
    id: str = Field(description="Video unit id, e.g. V07. Every video unit appears exactly once.")
    shows: str = Field(description="What is SEEN, in one sentence, including readable on-screen text and any "
                                   "object that matters (the product, the gift).")
    focus: str = Field(description="The ONE thing the viewer must see in this unit, in a few words "
                                   "('her face', 'the sweater', 'the price tag').")
    needs: List[str] = Field(description="EARLIER units (video or audio ids) this shot makes no sense without: a "
                                         "reaction needs what it reacts to; a reveal needs its setup. Only the "
                                         "DIRECT ones, usually one or none. Empty if the shot stands alone.")


class AudioUnit(BaseModel):
    id: str = Field(description="Audio unit id, e.g. A05. Every audio unit appears exactly once.")
    heard: str = Field(description="What is REALLY heard. Speech written out and corrected from the rough "
                                   "transcript (fix wrong words, add what it missed, leave out what it invented). "
                                   "Other sounds in round brackets: '(aaa)', '(door slams)', '(music)', '(silence)'.")
    needs: List[str] = Field(description="EARLIER units (audio or video ids) this sound makes no sense without: a "
                                         "punchline needs its setup line; 'look at this' needs the shot it points "
                                         "at. Only the DIRECT ones, usually one or none. Empty if it stands alone.")


class Units(BaseModel):
    summary: str = Field(description="One or two sentences: what this file is and the story it tells.")
    music: Music
    video_units: List[VideoUnit] = Field(description="Empty for an audio-only file.")
    audio_units: List[AudioUnit]


# =========================================================================== optional: the framing check
# Its own call, with its own stored answer (prep/framing_check.json). It does not touch call 1.
class ShotFraming(BaseModel):
    id: str = Field(description="Shot id, e.g. V07 (it is written in the top-left corner of the video). "
                                "Every shot appears exactly once.")
    fixed_focus_x: int = Field(description="-1 = the green line is on the right thing in this shot (the usual case). "
                                           "Otherwise: where the picture should be centred for the whole shot, left "
                                           "to right, as a percentage of the frame width (0 = left edge, 50 = "
                                           "middle, 100 = right edge).")
    why: str = Field(description="When you gave a position: what the line is on, and what it should be on. "
                                 "Empty string when the line is fine.")


class FramingCheck(BaseModel):
    shots: List[ShotFraming]


# =========================================================================== call 2: the edits
# One request per mode. Gemini sees call 1's stored descriptions (no video) and the requested lengths.
class SyncedEdit(BaseModel):
    target_s: float = Field(description="The length that was asked for.")
    segments: List[str] = Field(description="Segment ids (S01, S02, ...) to keep, in playing order. "
                                            "Segments not listed are cut.")
    story: str = Field(description="One or two sentences: how the story holds, and what was cut to get here.")


class SyncedEdits(BaseModel):
    edits: List[SyncedEdit] = Field(description="Exactly one per requested length.")


class CutawaySegment(BaseModel):
    audio: str = Field(description="The sound slot heard here, e.g. A04, or A05b for one part of a long sound.")
    video: str = Field(description="REQUIRED: the video unit (shot) SEEN while this sound plays, e.g. V03. Either a "
                                   "shot the sound was recorded over (its own picture), or a different shot (a cutaway).")


class CutawayEdit(BaseModel):
    target_s: float = Field(description="The length that was asked for.")
    segments: List[CutawaySegment] = Field(description="In playing order. Sound slots not listed are cut.")
    story: str = Field(description="One or two sentences: how the story holds, and what was cut to get here.")


class CutawayEdits(BaseModel):
    edits: List[CutawayEdit] = Field(description="Exactly one per requested length.")
