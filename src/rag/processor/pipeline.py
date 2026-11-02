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
