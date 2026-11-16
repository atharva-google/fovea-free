"""The output schemas of the two Gemini calls. Gemini is made to answer in exactly these shapes.

    call 1  Units      once per input FILE (stored): describes every unit of that file
    call 2  Timelines  text only: arranges units from one or more files into finished edits

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
    focus_x: int = Field(description="Where that thing is, left to right, as a percentage of the frame width: "
                                     "0 = left edge, 50 = middle, 100 = right edge.")
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


# =========================================================================== call 2: Timelines
class Mode(str, Enum):
    synced = "synced"    # every sound plays with the picture it was recorded with
    cutaway = "cutaway"  # a sound may play over a different picture


class Segment(BaseModel):
    audio: str = Field(description="The audio unit heard here, e.g. F1.A04. Empty string = the sound recorded "
                                   "with the picture in `video`.")
    video: str = Field(description="The video unit seen here, e.g. F1.V03. Empty string = the picture the sound in "
                                   "`audio` was recorded with. A video unit is used at most once per timeline. "
                                   "Never put someone else's speech over a face that is visibly talking.")


class Timeline(BaseModel):
    target_s: float = Field(description="The length that was asked for.")
    mode: Mode
    segments: List[Segment] = Field(description="In playing order. Anything not listed is cut.")
    music: str = Field(description="File id of a music track to play under the whole timeline, e.g. F3. "
                                   "Empty string = keep whatever music the sounds already carry.")
    story: str = Field(description="One or two sentences: how the story holds, and what was lost to get to this length.")


class Timelines(BaseModel):
    timelines: List[Timeline] = Field(description="Exactly one per requested length and mode.")
