import asyncio
import pytest

from rag.core.errors import NoTranscriptError, VideoUnavailableError
from rag.models import RawTranscript,VideoDocument
from rag.processor.segmenter import FixedWindowSegmenter
from rag.store import artifacts as A
from tests.conftest import VIDEO_ID,make_lines, make_meta

URL = f"https://youtu.be/{VIDEO_ID}?t=30"

def run(coro):
    return asyncio.run(coro)

def test_url_to_document(make_pipeline, store):
    pipeline, mf, cf = make_pipeline()
    stages = []

    async def progress(stage, p):
        stages.append((stage, p))

    doc = run(pipeline.run(URL, on_progress=progress))

    assert doc.video_id == VIDEO_ID
    assert doc.transcript_source == "captions_manual"
    assert doc.language == "en"
    assert len(doc.segments) > 5
    assert doc.chapters == [] and doc.synopsis == ""  # summarizer arrives in phase 3
    for name in (A.METADATA, A.RAW_TRANSCRIPT, A.SEGMENTS, A.DOCUMENT):
        assert store.exists(VIDEO_ID, name)
    assert store.load(VIDEO_ID, A.DOCUMENT, VideoDocument) == doc
    assert [s for s, _ in stages] == ["fetching", "segmenting", "summarizing"]
    assert [p for _, p in stages] == sorted(p for _, p in stages)


def test_rerun_resumes_from_artifacts(make_pipeline):
    pipeline, mf, cf = make_pipeline()
    first = run(pipeline.run(URL))
    second = run(pipeline.run(VIDEO_ID))
    assert first == second
    assert (mf.calls, cf.calls) == (1, 1)  # nothing re-fetched


def test_resume_after_crash_mid_pipeline(make_pipeline, store):
    pipeline, mf, cf = make_pipeline()
    run(pipeline.run(URL))
    # simulate a crash after fetching: later artifacts missing
    for name in (A.SEGMENTS, A.DOCUMENT):
        store.delete(VIDEO_ID, name)
    run(pipeline.run(URL))
    assert (mf.calls, cf.calls) == (1, 1)
    assert store.exists(VIDEO_ID, A.DOCUMENT)


def test_segmenter_change_triggers_resegmentation(make_pipeline, store):
    p1, _, _ = make_pipeline(segmenter=FixedWindowSegmenter(45, 10))
    d1 = run(p1.run(URL))
    p2, _, cf2 = make_pipeline(segmenter=FixedWindowSegmenter(20, 5))
    d2 = run(p2.run(URL))
    assert len(d2.segments) > len(d1.segments)
    assert cf2.calls == 0  # transcript reused, only segmentation redone


def test_no_captions_without_whisper_fails_permanently(make_pipeline, store):
    pipeline, _, _ = make_pipeline(lines=None)
    with pytest.raises(NoTranscriptError):
        run(pipeline.run(URL))
    assert store.exists(VIDEO_ID, A.METADATA)  # earlier stage kept
    assert not store.exists(VIDEO_ID, A.RAW_TRANSCRIPT)


def test_whisper_fallback_is_used_when_captions_missing(make_pipeline):
    reported = []

    class FakeTranscriber:
        def transcribe(self, meta, on_progress):
            for f in (0.25, 0.5, 1.0):
                on_progress(f)
            return RawTranscript(video_id=meta.video_id, language="en", source="whisper", lines=make_lines(50))

    async def progress(stage, p):
        reported.append(stage)

    pipeline, _, _ = make_pipeline(lines=None, transcriber=FakeTranscriber())
    doc = run(pipeline.run(URL, on_progress=progress))
    assert doc.transcript_source == "whisper"
    assert "transcribing" in reported


def test_too_long_video_is_rejected_before_captions(make_pipeline, store):
    pipeline, _, cf = make_pipeline(meta=make_meta(duration_s=10 * 3600))
    with pytest.raises(VideoUnavailableError):
        run(pipeline.run(URL))
    assert cf.calls == 0
    assert not store.exists(VIDEO_ID, A.METADATA)  # rejected metadata is not cached


def test_transcript_of_only_non_speech_fails(make_pipeline):
    from rag.models import TranscriptLine as L

    pipeline, _, _ = make_pipeline(lines=[L(start=0, end=5, text="[Music]"), L(start=5, end=9, text="♪♪")])
    with pytest.raises(NoTranscriptError):
        run(pipeline.run(URL))
