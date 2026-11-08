import asyncio
import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to hit real YouTube"),
]

# A long-lived, captioned public talk. Swap it if it ever disappears.
URL = os.getenv("LIVE_TEST_URL", "https://www.youtube.com/watch?v=kCc8FmEb1nY")


def test_metadata_and_captions(tmp_path):
    from rag.processor.fetcher import YouTubeCaptionFetcher, YtDlpMetadataFetcher, parse_video_id

    vid = parse_video_id(URL)
    meta = YtDlpMetadataFetcher().fetch(vid)
    assert meta.duration_s > 0 and meta.title
    raw = YouTubeCaptionFetcher().fetch(vid, ["en"])
    assert raw is not None and len(raw.lines) > 10


def test_full_pipeline(tmp_path, monkeypatch):
    from rag.core.config import Settings
    from rag.wiring import build_pipeline

    doc = asyncio.run(build_pipeline(Settings(data_dir=tmp_path)).run(URL))
    assert len(doc.segments) > 3
