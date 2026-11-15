""" chroma indexer,
    one collection per embedding model

 """

from __future__  import annotations

import logging
import re
from pathlib import Path
from rag.core.errors import TransientError, EmbeddingModelMismatchError
from rag.indexer.base import Embedder
from rag.models import Hit,VideoDocument

log=logging.getLogger(__name__)

COLLECTION_PREFIX ="segments__"

def collection_name(model_name: str)-> str:
    """ chroma names """
    slug =re.sub(r"[^a-z0-9._-]+", "-", model_name.lower()).strip("-._")
    return f"{COLLECTION_PREFIX}{slug}"[:500]

def make_client(*, host: str| None, port: int, persist_dir: Path| None):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    settings =ChromaSettings(anonymized_telemetry=False)
    if host:
        return chromadb.HttpClient(host=host,port=port,settings=settings)
    if persist_dir is None:
        return chromadb.EphemeralClient(settings=settings)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir),settings=settings)

class ChromaIndexer:
    def __init__(self,client, embedder: Embedder)-> None:
        self.embedder= embedder
        self._client =client
        name =collection_name(embedder.model_name)

        try:
            self._col =client.get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}},metadata={"embedding_model":embedder.model_name}, embedding_function=None,)

        except Exception as e:
            raise TransientError(f"Couldnt open the vector store: {e}") from e
        stored= (self._col.metadata or {}).get("embedding_model")
        if stored != embedder.model_name:
            raise EmbeddingModelMismatchError(
                f"Collection {name!r} was built with {stored!r}, but EMBEDDING_MODEL is "
                f"{embedder.model_name!r}. Run `videorag reindex` after changing models."
            )
        self._max_batch= self._batch_limit(client)

    @staticmethod
    def _batch_limit(client)-> int:
        try:
            return max(1, int(client.get_max_batch_size()))
        except Exception:
            return 1000
    @property
    def collection(self):
        return self._col

    def exists(self,video_id: str)-> bool:
        try:
            return bool(self._col.get(where={"video_id": video_id}, limit=1, include=[])["ids"])
        except Exception as e:
            raise TransientError(f"Vector store error: {e}") from e

    def index(self, doc:VideoDocument):
        segs= doc.segments
        vecs =self.embedder.embed_docs([s.text for s in segs])
        try:
            self._col.delete(where={"video_id": doc.video_id})
            for i in range(0,len(segs), self._max_batch):
                batch = segs[i : i + self._max_batch]
                self._col.add(
                    ids=[s.id for s in batch],
                    embeddings=vecs[i : i + self._max_batch],
                    documents=[s.text for s in batch],
                    metadatas=[
                        {"video_id": doc.video_id, "index": s.index, "start": s.start, "end": s.end}
                        for s in batch
                    ],
                )
            stored= len(self._col.get(where={"video_id": doc.video_id}, include=[]))["ids"]
        except Exception as e:
            raise TransientError(f"Vector store error while indexing {doc.video_id}: {e}") from e
        if stored !=len(segs):
            raise TransientError(f"Indexed {stored} of {len(segs)} segments for {doc.video_id}")
        log.info("indexed: %s: %d segments into %s",doc.video_id, stored,self._col.name)

    def delete(self, video_id: str) -> None:
        try:
            self._col.delete(where={"video_id": video_id})
        except Exception as e:
            raise TransientError(f"Vector store error: {e}") from e

    def search(self,query: str, k:int=5,video_id: str |None= None) -> list[Hit]:
        if k<=0:
            return []
        q= self.embedder.embed_query(query)

        try:
            res =self._col.query(
                query_embeddings=[q],
                n_results=k,
                where={"video_id": video_id} if video_id else None,
                include=["documents", "metadatas", "distances"],
            )

        except Exception as e:
            raise TransientError(f"Vector store error: {e}") from e
        hits = []
        for sid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(Hit(
                segment_id=sid, 
                video_id=str(meta["video_id"]), 
                index=int(meta["index"]), 
                start=float(meta["start"]), 
                end=float(meta["end"]), 
                text=text or "",
                score=min(max(1.0 - float(dist), 0.0), 1.0),))

        return hits
