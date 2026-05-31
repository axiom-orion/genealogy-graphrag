"""Dense ANN index. FAISS HNSW for the production path; transparent numpy fallback
so the repo runs even where faiss is unavailable (small corpora are fine either way)."""
from __future__ import annotations

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:  # pragma: no cover
    _HAS_FAISS = False


class VectorIndex:
    def __init__(self, dim: int, hnsw_m: int = 32, ef_search: int = 64):
        self.dim = dim
        self.backend = "faiss-hnsw" if _HAS_FAISS else "numpy"
        self._faiss = None
        self._mat: np.ndarray | None = None
        self._hnsw_m, self._ef_search = hnsw_m, ef_search

    def build(self, embeddings: np.ndarray) -> VectorIndex:
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        if _HAS_FAISS:
            # inner product on normalised vectors == cosine similarity
            idx = faiss.IndexHNSWFlat(self.dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efSearch = self._ef_search
            idx.add(emb)
            self._faiss = idx
        else:
            self._mat = emb
        return self

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=False) if i != -1]
        sims = (self._mat @ q[0])
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]
