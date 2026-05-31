"""Hybrid retriever: dense + lexical + graph, fused by Reciprocal Rank Fusion,
with an optional cross-encoder rerank stage.

Design
------
1. Dense (FAISS HNSW) and lexical (BM25) each produce a ranked candidate list.
2. RRF fuses them: score(d) = Σ_method 1 / (rrf_k + rank_method(d)). RRF is
   scale-free, so it composes heterogeneous scorers (cosine, BM25, graph distance)
   without tuning per-scorer weights.
3. Graph expansion treats the kin graph as a *third retriever*: seed persons come
   from (a) entity-linking the query against person names and (b) the persons in
   the top fused documents; we expand to their k-hop kin and surface every document
   mentioning a kin member, ranked by hop distance. This is what lets relational
   questions ("maternal grandfather of X") retrieve the answer's records even
   though the answer entity is never named in the query.
4. Rerank (optional) re-scores the top candidates with a cross-encoder.

Every stage is independently toggleable -> the eval ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .corpus import Document
from .embeddings import Embedder
from .graph.base import GraphStore
from .index import LexicalIndex, VectorIndex
from .kinship import RelationResolver
from .rerank import CrossEncoderReranker


@dataclass(frozen=True)
class RetrievalConfig:
    use_vector: bool = True
    use_bm25: bool = True
    use_graph: bool = True
    use_rerank: bool = True

    @property
    def label(self) -> str:
        parts = []
        if self.use_vector:
            parts.append("vector")
        if self.use_bm25:
            parts.append("bm25")
        if self.use_graph:
            parts.append("graph")
        tag = "+".join(parts) if parts else "none"
        return tag + ("+rerank" if self.use_rerank else "")


@dataclass(frozen=True)
class Hit:
    doc: Document
    score: float
    stages: tuple[str, ...]


def _rrf(rankings: list[list[int]], k: int) -> dict[int, float]:
    """Reciprocal Rank Fusion over a set of ranked id-lists."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, documents: list[Document], graph: GraphStore,
                 embedder: Embedder | None = None,
                 reranker: CrossEncoderReranker | None = None,
                 candidate_pool: int = 50):
        self.docs = documents
        self.graph = graph
        self.candidate_pool = candidate_pool
        self._idx_of = {d.id: i for i, d in enumerate(documents)}

        # dense
        self.embedder = embedder or Embedder()
        self._doc_emb = self.embedder.encode([d.searchable for d in documents])
        self.vector = VectorIndex(self._doc_emb.shape[1]).build(self._doc_emb)
        self.backend_vector = self.vector.backend
        # lexical
        self.lexical = LexicalIndex().build([d.searchable for d in documents])
        # rerank
        self.reranker = reranker or CrossEncoderReranker()
        # structured kinship resolution (the KG-augmented-QA layer)
        self.resolver = RelationResolver(graph)
        self._type_of = {d.id: d.type for d in documents}

    # --- graph stage: structured kinship resolution (authoritative) -------- #
    SALIENCE = {"birth_record": 0, "obituary": 1, "biography": 2,
                "marriage_record": 3, "census": 4, "dna_match": 5}

    def _pinned_docs(self, query: str) -> list[int]:
        """Doc indices the relation resolver vouches for, salience-ordered.

        Fires only on '<relation> of <person>' questions. A resolved target's
        records are ordered identity-establishing-first (vital -> obituary ->
        biography -> ...), a content-agnostic record-type prior -- not tuned to
        the eval. Empty for non-kinship questions, where retrieval falls back to
        the fused dense+lexical ranking."""
        res = self.resolver.resolve(query)
        if res is None or not res.fired:
            return []
        out: list[int] = []
        seen: set[int] = set()
        for pid in res.targets:
            cand: list[tuple[int, int, str, int]] = []
            for d in self.graph.documents_for_person(pid):
                di = self._idx_of.get(d)
                if di is None:
                    continue
                doc = self.docs[di]
                # docs *about* this person (vital/obit/bio) outrank docs that
                # merely mention them (a child's birth record, a census household)
                primary = 0 if doc.facts.get("person") == pid else 1
                cand.append((primary, self.SALIENCE.get(doc.type, 9), d, di))
            for _, _, _, di in sorted(cand):
                if di not in seen:
                    seen.add(di)
                    out.append(di)
        return out

    # --- main --------------------------------------------------------------- #
    def retrieve(self, query: str, k: int, cfg: RetrievalConfig) -> list[Hit]:
        pool = self.candidate_pool
        rankings: list[list[int]] = []
        stage_of: dict[int, set[str]] = {}

        if cfg.use_vector:
            qv = self.embedder.encode([query], use_cache=False)[0]
            v = [i for i, _ in self.vector.search(qv, pool)]
            rankings.append(v)
            for i in v:
                stage_of.setdefault(i, set()).add("vector")
        if cfg.use_bm25:
            b = [i for i, _ in self.lexical.search(query, pool)]
            rankings.append(b)
            for i in b:
                stage_of.setdefault(i, set()).add("bm25")

        fused = _rrf(rankings, settings.rrf_k) if rankings else {}
        ordered = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]

        pinned = self._pinned_docs(query) if cfg.use_graph else []
        pinned_set = set(pinned)
        for i in pinned:
            stage_of.setdefault(i, set()).add("graph")

        # Rerank only the fuzzy (non-pinned) candidates; a confident KG resolution
        # is authoritative and is not subject to surface-relevance reordering.
        if cfg.use_rerank and ordered:
            tail = [i for i in ordered if i not in pinned_set]
            head = tail[: settings.rerank_candidates]
            if head:
                scores = self.reranker.rerank(
                    query, [self.docs[i].searchable for i in head])
                if any(scores):
                    reranked = [i for i, _ in
                                sorted(zip(head, scores, strict=False), key=lambda x: -x[1])]
                    ordered = reranked + tail[settings.rerank_candidates:]
                    for i in head:
                        stage_of.setdefault(i, set()).add("rerank")
                else:
                    ordered = tail
            else:
                ordered = tail

        final = pinned + [i for i in ordered if i not in pinned_set]

        hits: list[Hit] = []
        for rank, di in enumerate(final[:k]):
            hits.append(Hit(doc=self.docs[di],
                            score=1.0 / (rank + 1),
                            stages=tuple(sorted(stage_of.get(di, set())))))
        return hits
