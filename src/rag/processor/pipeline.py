from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Awaitable
from typing import Protocol

from rag.core.errors import NoTranscriptError
from rag.models import (
    Chapter, RawTranscript,SegmentArtifact, VideoDocument,VideoMetadata, stage_progress
)
from rag.processor.cleaner import clean
from rag.processor.fetcher import CaptionFetcher, MetadataFetcher,parse_video_id, validate_metadata
from rag.processor.transcriber import Transcriber
from rag.store import artifacts as A
from rag.store.artifacts import ArtifactStore
from rag.processor.segmenter import Segmenter

log =logging.getLogger(__name__)

Progressfn= Callable[[str,float], Awaitable[None]]

async def _no_progress(stage: str,progress:float)-> None:
    return None

class Summary(Protocol):
    chapters: list[Chapter]
    synopsis: str

class Summarizer(Protocol):
    """ returns chapters with  timestamps and synopsis"""
    async def summarize(self, meta: VideoMetadata, segements: SegmentArtifact)-> Summary: ...

class Pipeline():
    def __init__(self, *,metadata_fetcher:MetadataFetcher, caption_fetcher: CaptionFetcher, segmenter: Segmenter, artifacts: ArtifactStore, preferred_languages:list[str], max_video_duration_s:int, transcriber: Transcriber| None=None, summarizer: Summarizer| None=None) -> None:
        self.metadata_fetcher= metadata_fetcher
        self.caption_fetcher = caption_fetcher
        self.segmenter =segmenter
        self.artifacts = artifacts
        self.preferred_languages = preferred_languages
        self.max_video_duration_s = max_video_duration_s
        self.transcriber = transcriber
        self.summarizer = summarizer

    async def run(self, url_or_id: str,on_progress: Progressfn= _no_progress)-> VideoDocument:
        video_id =parse_video_id(url_or_id)
        
        await on_progress("fetching", stage_progress("fetching"))
        meta =await self._metadata(video_id)
        raw = await self._transcript(meta, on_progress)

        await on_progress("segmenting", stage_progress("segmenting"))
        segs =await asyncio.to_thread(self._segments, video_id, raw)

        await on_progress("summarizing",stage_progress("summarizing"))
        chapters,synopsis =await self._summary(meta,segs)

        doc = VideoDocument(
            video_id=video_id,
            url=meta.url,
            title=meta.title,
            channel=meta.channel,
            duration_s=meta.duration_s,
            language=raw.language or meta.language or "und",
            transcript_source=raw.source,
            segments=segs.segments,
            chapters=chapters,
            synopsis=synopsis,
        )

        self.artifacts.save(video_id, A.DOCUMENT, doc)
        log.info("document ready for %s: %d segments, %d chapters", video_id, len(doc.segments), len(chapters))
        return doc


    async def _metadata(self, video_id: str)-> VideoMetadata:
        meta = self.artifacts.load_if_exists(video_id, A.METADATA, VideoMetadata)
        if meta is None:
            meta =await asyncio.to_thread(self.metadata_fetcher.fetch,video_id)
            validate_metadata(meta, self.max_video_duration_s)
            self.artifacts.save(video_id, A.METADATA,meta)
        return meta

    # cool stufff
    async def _transcript(self, meta: VideoMetadata, on_progress: Progressfn) -> RawTranscript:
        vid = meta.video_id
        raw = self.artifacts.load_if_exists(vid, A.RAW_TRANSCRIPT, RawTranscript)
        if raw is not None:
            return raw

        raw= await asyncio.to_thread(self.caption_fetcher.fetch,vid, self.preferred_languages)
        if raw is None or not raw.lines:
            if self.transcriber is None:
                raise NoTranscriptError("This video has no captions, and whisper fallback is not enabled yet")

            await on_progress("transcribing",stage_progress("transcribing"))
            loop= asyncio.get_running_loop()

            # called from woerker thread
            def report(fraction:float)-> None:
                asyncio.run_coroutine_threadsafe(on_progress("transcribing",stage_progress("transcribing",fraction)), loop)
            
            raw =await asyncio.to_thread(self.transcriber.transcribe,meta, report)
            if not raw.lines:
                raise NoTranscriptError("Whisper produced an empty transcript (the video may have no speech).")

        self.artifacts.save(vid,A.RAW_TRANSCRIPT, raw)
        return raw

    def _segments(self, video_id: str, raw: RawTranscript) -> SegmentArtifact:
        existing = self.artifacts.load_if_exists(video_id, A.SEGMENTS, SegmentArtifact)
        if existing is not None and existing.segmenter == self.segmenter.name:
            return existing
        if existing is not None:
            log.info("segmenter changed (%s -> %s); re-segmenting %s", existing.segmenter, self.segmenter.name, video_id)
            self.artifacts.delete(video_id, A.SUMMARY)  # chapters point at old segment indices

        lines = clean(raw.lines)
        if not lines:
            raise NoTranscriptError("The transcript is empty after removing non-speech markers.")
        segs = SegmentArtifact(video_id=video_id,segmenter=self.segmenter.name,segments=self.segmenter.segment(video_id, lines),)
        self.artifacts.save(video_id, A.SEGMENTS, segs)
        return segs

    async def _summary(self, meta: VideoMetadata, segs: SegmentArtifact) -> tuple[list[Chapter], str]:
        if self.summarizer is None:
            return [], ""
        
        result = await self.summarizer.summarize(meta, segs)
        return list(result.chapters), result.synopsis