from typing import Literal
from pydantic import BaseModel, Field

class Hit(BaseModel):
    segment_id: str
    video_id :str
    index:int
    start: float
    end: float
    text: str
    score: float

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AskRequest(BaseModel):
    question:str= Field(min_length=1,max_length=2000)
    history: list[Turn] =Field(default_factory=list)

class Citation(BaseModel):
    ref: int
    start: float
    end: float
    text: str

class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    covered: bool