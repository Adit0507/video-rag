""" indexer and embedder protocols """

from __future__ import annotations
from typing import Protocol
from rag.models import Hit, VideoDocument

class Embedder(Protocol):
    @property
    def model_name(self)-> str:
        """stored with vectors"""
        ...

    def embed_docs(self, texts:list[str])-> list[list[float]]:
        """Unit-length vectors for segment texts."""
        ...
    def embed_query(self, text: str) -> list[float]:
        """Unit-length vector for a search query (applies the model's query prefix)."""
        ...

class Indexer(Protocol):
    def exists(self, video_id: str) ->bool: ...

    def index(self,doc: VideoDocument)-> None:
        ...
    def delete(self, video_id: str)->None: ...

    def search(self, query: str, k: int = 5, video_id: str | None = None) -> list[Hit]:
        ...