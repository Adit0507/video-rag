from typing import Literal
from datetime import datetime
from pydantic import BaseModel

JobStatus= Literal["queued", "fetching", "transcribing", "segmenting", "summarizing", "indexing", "done", "failed"]
ACTIVE_STATUSES: frozenset[str]=frozenset({"queued", "fetching", "transcribing", "segmenting", "summarizing", "indexing"})
STAGE_START: dict[str, float] = {
    "queued": 0.0,
    "fetching": 0.02,
    "transcribing": 0.10,
    "segmenting": 0.50,
    "summarizing": 0.55,
    "indexing": 0.85,
    "done": 1.0,
}

def stage_progress(stage:str, fraction: float=0.0)-> float:
    order =list(STAGE_START)
    start =STAGE_START[stage]
    nxt = STAGE_START[order[order.index(stage) + 1]] if stage != "done" else 1.0
    return round(start +(nxt- start)*min(max(fraction, 0.0), 1.0), 4)

class Job(BaseModel):
    job_id: str
    video_id: str
    url:str
    status:JobStatus
    progress: float= 0.0
    error: str| None=None
    created_at: datetime
    updated_at: datetime
    