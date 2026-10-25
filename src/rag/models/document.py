from typing import Literal
from pydantic import BaseModel, model_validator,Field

TranscriptSource= Literal["captions_manual","captions_auto", "whisper"]

class TranscriptLine(BaseModel):
    """One caption cue """
    start:float= Field(ge=0)
    end: float =Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _end_after_start(self)-> "TranscriptLine":
        if self.end <self.start:
            self.end= self.start
        return self

class VideoMetadata(BaseModel):
    video_id: str
    url:str
    title: str
    channel: str
    duration_s: float
    language: str| None=None
    is_live: bool=False
    age_limit: int =0
    availability: str |None=None

class RawTranscript(BaseModel):
    """transcript exactly as fetched"""
    video_id: str
    language: str
    source: TranscriptSource
    lines: list[TranscriptLine]

class Segment(BaseModel):
    id: str
    index: int
    start: float
    end: float
    text: str

class SegmentArtifact(BaseModel):
    """segmenter records the config"""
    video_id: str
    segmenter:str 
    segments: list[Segment]

class Chapter(BaseModel):
    title: str
    start:float
    end: float
    summary: str

class VideoDocument(BaseModel):    
    schema_version:int =2
    video_id: str
    url: str
    title: str
    channel: str
    duration_s: float
    language: str
    transcript_source: TranscriptSource
    segments: list[Segment]
    chapters: list[Chapter] =Field(default_factory=list)
    synopsis: str=""