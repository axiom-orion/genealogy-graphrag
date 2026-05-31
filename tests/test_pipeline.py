"""Test suite.

Fast checks (corpus, graph, resolver) run with no model load. The retrieval
property tests share one session-scoped pipeline so the embedding model is loaded
once. Run: pytest -q
"""
from __future__ import annotations

import pytest

from genealogy_rag import GenealogyRAG, RetrievalConfig, load_documents, load_questions
from genealogy_rag.graph import NetworkXGraphStore
from genealogy_rag.kinship import RelationResolver


# --------------------------- fixtures -------------------------------------- #
@pytest.fixture(scope="session")
def graph() -> NetworkXGraphStore:
    return NetworkXGraphStore().load()


@pytest.fixture(scope="session")
def rag() -> GenealogyRAG:
    return GenealogyRAG()


# --------------------------- corpus + graph -------------------------------- #
def test_corpus_loads():
    docs = load_documents()
    assert len(docs) > 50
    assert all(d.id and d.text and d.citation for d in docs)


def test_questions_have_resolvable_gold():
    docs = {d.id for d in load_documents()}
    qs = load_questions()
    assert len(qs) > 20
    for q in qs:
        assert q.relevant_doc_ids, f"{q.id} has no gold docs"
        assert set(q.relevant_doc_ids) <= docs, f"{q.id} references unknown doc"


def test_graph_kinship_navigation(graph):
    # P21 Nancy Ainsworth -> mother P12 Alice -> father P05 Thomas
    assert "P05" in graph.parents("P12")
    assert "P12" in graph.parents("P21")
    assert graph.sex("P05") == "M"
    assert graph.shortest_path("P21", "P05")[0] == "P21"


# --------------------------- relation resolver ----------------------------- #
def test_resolver_resolves_known_relations(graph):
    r = RelationResolver(graph)
    res = r.resolve("Who was the maternal grandfather of Nancy Ainsworth?")
    assert res and res.fired and res.targets == ["P05"]  # Thomas Calloway

    res = r.resolve("Who was the paternal grandfather of David Calloway?")
    assert res and res.targets == ["P05"]


@pytest.mark.parametrize("query", [
    "Which children were enumerated in the household of Thomas Calloway in the 1910 census?",
    "Which source record documents the marriage of Thomas Calloway and Margaret Brandt?",
    "In what year was Thomas Calloway born?",
])
def test_resolver_does_not_misfire(graph, query):
    """Resolver must stay silent on non-kinship questions so they fall back to
    fuzzy retrieval rather than being hijacked by a spurious graph answer."""
    r = RelationResolver(graph)
    res = r.resolve(query)
    assert res is None or not res.fired


# --------------------------- retrieval property ---------------------------- #
def _recall_at_k(rag, query, gold, k, cfg):
    hits = [h.doc.id for h in rag.retrieve(query, k=k, cfg=cfg)]
    return len(set(hits[:k]) & set(gold)) / len(gold)


def test_graph_lifts_relational_recall(rag):
    """Central claim: structured graph resolution recovers relational questions
    that pure dense+lexical retrieval cannot answer."""
    q = next(x for x in load_questions() if x.category == "relational")
    hybrid = RetrievalConfig(use_vector=True, use_bm25=True,
                             use_graph=False, use_rerank=False)
    graph = RetrievalConfig(use_vector=True, use_bm25=True,
                            use_graph=True, use_rerank=False)
    r_hybrid = _recall_at_k(rag, q.question, q.relevant_doc_ids, 5, hybrid)
    r_graph = _recall_at_k(rag, q.question, q.relevant_doc_ids, 5, graph)
    assert r_graph > r_hybrid
    assert r_graph == 1.0


def test_lookup_unaffected_by_graph(rag):
    """Graph resolution must not degrade questions it does not fire on."""
    q = next(x for x in load_questions() if x.category == "lookup")
    hybrid = RetrievalConfig(True, True, False, False)
    graph = RetrievalConfig(True, True, True, False)
    assert _recall_at_k(rag, q.question, q.relevant_doc_ids, 5, graph) >= \
        _recall_at_k(rag, q.question, q.relevant_doc_ids, 5, hybrid)
