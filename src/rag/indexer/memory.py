"""in memory indexer """

from __future__ import annotations
from rag.indexer.base import Embedder
from rag.models import Hit,Segment, VideoDocument

class InMemoryIndexer:
    def __init__(self, embedder: Embedder)-> None:
        self.embedder= embedder
        self._rows: dict[str, list[tuple[Segment, list[float]]]] ={}

    def exists(self, video_id: str)-> bool:
        return bool(self._rows.get(video_id))

    def index(self, doc: VideoDocument) ->None:
        vecs =self.embedder.embed_docs([s.text for s in doc.segments])
        self._rows[doc.video_id] =list(zip(doc.segments, vecs))

    def delete(self, video_id: str) -> None:
        self._rows.pop(video_id, None)

    def search(self,query:str, k: int=5, video_id: str |None =None) ->list[Hit]:
        if k<= 0:
            return []

        q= self.embedder.embed_query(query)
        vids= [video_id] if video_id is not None else list(self._rows)
        scored =[]

        for v in vids:
            for seg, vec in self._rows.get(v, []):
                sim =sum(a * b for a, b in zip(q, vec))
                scored.append((sim, v,seg))

        scored.sort(key=lambda x: (-x[0],x[1],x[2].index))

        return [
            Hit(segment_id=seg.id, video_id=v, index=seg.index,start=seg.start, end=seg.end,text=seg.text,score=min(max(sim,0.0), 1.0),)
            for sim, vid, seg in scored[:k]
        ]
