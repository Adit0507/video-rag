""" API,worker, CLI and eval runner all build their objects here, so they can never disagreee about which 
llm,indexer or segmenter is in usee """

from __future__ import annotations
from rag.core.config import Settings
from rag.processor.fetcher import YTCaptionFetcher,YtDlpMetadataFetcher
from rag.processor.pipeline import Pipeline
from rag.processor.segmenter import FixedWindowSegmenter
from rag.store.artifacts import ArtifactStore

def build_artifacts(settings:Settings)-> ArtifactStore:
    return ArtifactStore(settings.data_dir)

def build_segmenter(settings: Settings)-> FixedWindowSegmenter:
    return FixedWindowSegmenter(window_s=settings.segment_window_s, overlap_s=settings.segment_overlap_s,max_chars=settings.segment_max_chars)

def build_pipeline(settings: Settings)-> Pipeline:
    return Pipeline(
        metadata_fetcher=YtDlpMetadataFetcher(settings.youtube_proxy_url, settings.youtube_cookies_file),
        caption_fetcher=YTCaptionFetcher(settings.youtube_proxy_url),
        segmenter=build_segmenter(settings),
        artifacts=build_artifacts(settings),
        preferred_languages=settings.preferred_languages,
        max_video_duration_s=settings.max_video_duration_s,
        transcriber=None,
        summarizer=None,
        )
