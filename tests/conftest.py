from __future__ import annotations
import random
import pytest

from rag.models import RawTranscript, TranscriptLine,VideoMetadata
from rag.processor.fetcher import canonical_url
from rag.processor.pipeline import Pipeline
from rag.processor.segmenter import FixedWindowSegmenter
from rag.store.artifacts import ArtifactStore

VIDEO_ID= "dQw4w9WgXcQ"
DEFAULT = object() 

WORDS=(
    "the model learns attention weights over every token in the sequence and then we "
    "look at how gradients flow back through the network during training which is why "
    "normalization matters so much for stability"
).split()

def make_lines(n: int=200,seed: int =0, step: float= 3.0) ->list[TranscriptLine]:
    rng= random.Random(seed)
    lines,t =[], 0.0
    for _ in range(n):
        k=rng.randint(4,12)
        start = rng.randint(0, len(WORDS) - k)
        text = " ".join(WORDS[start : start + k])
        lines.append(TranscriptLine(start=round(t,2), end=round(t +step+1.5, 2), text=text))
        t += step * rng.uniform(0.7, 1.3)
    return lines

def make_meta(video_id: str = VIDEO_ID, duration_s: float = 600.0, **kw) -> VideoMetadata:
    return VideoMetadata(
        video_id=video_id,
        url=canonical_url(video_id),
        title="Test video",
        channel="Test channel",
        duration_s=duration_s,
        language="en",
        **kw,
    )

class FakeMetdataFetcher:
    def __init__(self, meta: VideoMetadata|None=None)->None:
        self.meta =meta or make_meta()
        self.calls =0

    def fetch(self, video_id: str)-> VideoMetadata:
        self.calls +=1
        return self.meta.model_copy(update={"video_id": video_id, "url": canonical_url(video_id)})

class FakeCaptionFetcher:
    def __init__(self, lines: list[TranscriptLine] | None, source: str = "captions_manual") -> None:
        self.lines = lines
        self.source = source
        self.calls = 0

    def fetch(self, video_id: str, preferred_languages: list[str]) -> RawTranscript | None:
        self.calls += 1
        if self.lines is None:
            return None
        return RawTranscript(video_id=video_id, language="en", source=self.source, lines=self.lines)


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "data")

@pytest.fixture
def make_pipeline(store):
    def _make(
        lines=DEFAULT,
        meta: VideoMetadata | None = None,
        segmenter: FixedWindowSegmenter | None = None,
        transcriber=None,
        max_duration: int = 4 * 3600,
    ):
        mf=FakeMetdataFetcher(meta)
        cf = FakeCaptionFetcher(make_lines() if lines is DEFAULT else lines)
        p = Pipeline(
            metadata_fetcher=mf,
            caption_fetcher=cf,
            segmenter=segmenter or FixedWindowSegmenter(),
            artifacts=store,
            preferred_languages=["en"],
            max_video_duration_s=max_duration,
            transcriber=transcriber,
        )
        return p, mf, cf
    return _make