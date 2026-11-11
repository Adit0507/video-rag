"""  """
from __future__ import annotations

import logging
import math
import re
import threading
import zlib
from pathlib import Path

log = logging.getLogger(__name__)

_BGE_EN_QUERY = "Represent this sentence for searching relevant passages: "
def default_prefixes(model_name: str)-> tuple[str,str]:
    name= model_name.lower()
    if "bge"in name and "-en" in name:
        return _BGE_EN_QUERY, ""
    if "e5" in name:  
        return "query: ", "passage: "
    return "", ""


class SentenceTransformerEmbedder:
    def __init__(self,
                model_name: str,
                *, cache_dir: Path|None= None,
                batch_size:int =32,
                query_prefix: str | None = None,
                document_prefix: str | None = None,
                ) ->None:
        self._name =model_name
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        dq,dd =default_prefixes(model_name)
        self._query_prefix =dq if query_prefix is None else query_prefix
        self._doc_prefix = dd if document_prefix is None else document_prefix
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_name(self)-> str:
        return self._name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("loading embedding model %s",self._name)
            self._model =SentenceTransformer(self._name, device="cpu",cache_folder=str(self._cache_dir) if self._cache_dir else None,)
        
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            model = self._load()
            vecs = model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return [list(map(float, v)) for v in vecs]

    def embed_docs(self, texts: list[str])-> list[list[float]]:
        return self._encode([self._doc_prefix] +t for t in texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

class HashEmbedder:

    def __init__(self,dim:int=512)-> None:
        self.dim= dim

    @property
    def model_name(self) -> str:
        return f"hash-{self.dim}"

    def _vec(self, text:str)-> list[float]:
        toks= _TOKEN_RE.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v=[0.0] *self.dim
        for f in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0

        norm = math.sqrt(sum(x * x for x in v))
        if norm== 0:
            v[0] =1.0
            return v
        return [x / norm for x in v]

    def embed_docs(self, texts: list[str]) ->list[list[float]]:
        return [self._vec(t) for t in texts]
    def embed_query(self, text:str)-> list[float]:
        return self._vec(text)