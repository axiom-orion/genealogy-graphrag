"""End-to-end facade: load corpus + graph + indexes once, answer queries with
cited sources. The eval scores the retrieval stage directly via `retrieve()`."""
from __future__ import annotations

from .corpus import Document, load_documents
from .graph import load_graph_store
from .provenance import GroundedAnswer, build_citations
from .retrieval import Hit, HybridRetriever, RetrievalConfig


class GenealogyRAG:
    def __init__(self, documents: list[Document] | None = None):
        self.documents = documents or load_documents()
        self.graph = load_graph_store().load()
        self.retriever = HybridRetriever(self.documents, self.graph)

    def retrieve(self, query: str, k: int = 5,
                 cfg: RetrievalConfig | None = None) -> list[Hit]:
        return self.retriever.retrieve(query, k, cfg or RetrievalConfig())

    def answer(self, query: str, k: int = 5,
               cfg: RetrievalConfig | None = None) -> GroundedAnswer:
        hits = self.retrieve(query, k, cfg)
        docs = [h.doc for h in hits]
        return GroundedAnswer(query=query, context=docs,
                              citations=build_citations(docs))
