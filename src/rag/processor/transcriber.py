""" whisper fallback for vids without captions """

from __future__ import annotations
from collections.abc import Callable
from typing import Protocol
from rag.models import RawTranscript,VideoMetadata

TranscribeProgress= Callable[[float], None]

class Transcriber(Protocol):
    def transcribe(self, meta:VideoMetadata, on_progress: TranscribeProgress)-> RawTranscript: ...