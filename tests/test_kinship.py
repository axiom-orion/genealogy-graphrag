"""Relation-resolver edge cases (no model load; runs in well under a second).

The resolver's output is treated as authoritative -- retrieval pins the targets'
records above everything else -- so these tests pin both halves of its contract:
it resolves the phrasings people actually use, and it abstains (None / not fired)
whenever a resolution could be confidently wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from genealogy_rag import load_questions
from genealogy_rag.graph import NetworkXGraphStore
from genealogy_rag.kinship import MAX_QUERY_CHARS, RelationResolver


@pytest.fixture(scope="module")
def graph() -> NetworkXGraphStore:
    return NetworkXGraphStore().load()


@pytest.fixture(scope="module")
def resolver(graph) -> RelationResolver:
    return RelationResolver(graph)


def names(graph, res) -> list[str] | None:
    """Sorted target names, or None when the resolver abstained or found nobody."""
    if res is None or not res.fired:
        return None
    return sorted(graph.person_name(t) for t in res.targets)


# --------------------------- resolves --------------------------------------- #
@pytest.mark.parametrize("query, expected", [
    # the gold phrasing still works
    ("Who was the paternal grandfather of David Calloway?", ["Thomas Calloway"]),
    # possessive, straight and curly apostrophe
    ("Who was David Calloway's paternal grandfather?", ["Thomas Calloway"]),
    ("Who was David Calloway’s paternal grandfather?", ["Thomas Calloway"]),
    # synonyms
    ("Who was the dad of David Calloway?", ["Robert Calloway"]),
    ("Who was the paternal grandpa of David Calloway?", ["Thomas Calloway"]),
    # chains, both forms, and mixed
    ("Who was the father of the father of David Calloway?", ["Thomas Calloway"]),
    ("Who was David Calloway's father's father?", ["Thomas Calloway"]),
    ("Who was the father of David Calloway's father?", ["Thomas Calloway"]),
    # "great" counts generations; paternal = through the father, so both of his
    # grandfathers are paternal great-grandfathers
    ("Who was the great-grandfather of David Calloway?", ["Henry Brandt", "Josiah Calloway"]),
    ("Who was the paternal great-grandfather of David Calloway?",
     ["Henry Brandt", "Josiah Calloway"]),
    # the possessive chain ends at a non-kin word, keeping the kin part
    ("When was David Calloway's father born?", ["Robert Calloway"]),
    # entity linking: a one-edit typo, and a first name that is unique in the graph
    ("Who was the paternal grandfather of David Caloway?", ["Thomas Calloway"]),
    ("Who was the paternal grandfather of David?", ["Thomas Calloway"]),
    # descendants
    ("Who were the grandchildren of Josiah Calloway?",
     ["Alice Calloway", "Clara Mercer", "James Calloway", "Robert Calloway"]),
    # a fuzzy anchor carries its possessive chain: the longest linked span wins
    ("Who was the mother of David Caloway's father?", ["Margaret Brandt"]),
])
def test_resolves(graph, resolver, query, expected):
    assert names(graph, resolver.resolve(query)) == expected


def test_chain_records_its_path(resolver):
    res = resolver.resolve("Who was the father of the father of David Calloway?")
    assert res is not None
    assert [s.relation for s in res.path] == ["father", "father"]
    assert res.relation == "father" and res.linked_by == "exact"


def test_links_are_labelled(resolver):
    typo = resolver.resolve("Who was the paternal grandfather of David Caloway?")
    first = resolver.resolve("Who was the paternal grandfather of David?")
    assert typo is not None and typo.linked_by == "fuzzy"
    assert first is not None and first.linked_by == "first_name"


# --------------------------- abstains ---------------------------------------- #
@pytest.mark.parametrize("query", [
    # a generation the graph doesn't have: no answer, not the nearest generation
    "Who was the great-great-grandfather of David Calloway?",
    # negation and exclusion
    "Who was not the paternal grandfather of David Calloway?",
    "Who wasn't the paternal grandfather of David Calloway?",
    "Which grandfather of David Calloway, other than Thomas, is recorded?",
    # relations the graph doesn't model
    "Who was the uncle of David Calloway?",
    "Who was David Calloway's uncle?",
    "Who was David Calloway's maternal uncle?",
    "Who was David Calloway's father-in-law?",
    "Who was the father-in-law of David Calloway?",
    "Who was the half brother of David Calloway?",
    "Who was the father of the first wife of David Calloway?",
    # modifiers the relation can't take, or that contradict each other
    "Who was the maternal father of David Calloway?",
    "Who was the great brother of David Calloway?",
    "Who was the maternal paternal grandfather of David Calloway?",
    # more than one anchor in a relation position
    "Who were the father of David Calloway and the mother of Ruth Calloway?",
    # the original non-kinship guards
    "Which children were enumerated in the household of Thomas Calloway in the 1910 census?",
    "Which source record documents the marriage of Thomas Calloway and Margaret Brandt?",
    "In what year was Thomas Calloway born?",
    "What did David Calloway's 1910 census entry record?",
])
def test_abstains(graph, resolver, query):
    assert names(graph, resolver.resolve(query)) is None


def test_query_length_cap(graph, resolver):
    base = "Who was the paternal grandfather of David Calloway?"
    padded = base + " " * (MAX_QUERY_CHARS - len(base))  # exactly at the cap: still parsed
    assert names(graph, resolver.resolve(padded)) == ["Thomas Calloway"]
    assert resolver.resolve(padded + " ") is None  # one past the cap: abstains


# --------------------------- ambiguous anchors -------------------------------- #
def _tiny_graph(tmp_path: Path) -> NetworkXGraphStore:
    """Two Johns, and two near-identical surnames, so linking has to choose."""
    persons = [
        {"id": "A", "name": "John Harrow", "sex": "M"},
        {"id": "B", "name": "John Barrow", "sex": "M"},
        {"id": "C", "name": "Mary Harrow", "sex": "F"},
        {"id": "D", "name": "Peter Harrow", "sex": "M"},
    ]
    rels = [
        {"src": "D", "dst": "A", "rel": "CHILD_OF"},
        {"src": "A", "dst": "D", "rel": "PARENT_OF"},
    ]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"persons": persons, "relationships": rels, "person_documents": {}}))
    return NetworkXGraphStore().load(path)


def test_ambiguous_first_name_abstains(tmp_path):
    r = RelationResolver(_tiny_graph(tmp_path))
    assert r.resolve("Who was the son of John?") is None  # two Johns


def test_ambiguous_fuzzy_name_abstains(tmp_path):
    r = RelationResolver(_tiny_graph(tmp_path))
    # "john darrow" is one edit from both "john harrow" and "john barrow"
    assert r.resolve("Who was the son of John Darrow?") is None


def test_unique_fuzzy_name_links(tmp_path):
    g = _tiny_graph(tmp_path)
    res = RelationResolver(g).resolve("Who was the son of John Harow?")
    assert res is not None and res.fired and res.targets == ["D"]


# --------------------------- gold set unchanged ------------------------------ #
def test_gold_relational_questions_resolve_to_their_answer(resolver):
    for q in load_questions():
        res = resolver.resolve(q.question)
        if q.category == "relational":
            assert res is not None and res.fired, q.id
            assert q.answer_entity in res.targets, q.id
        else:
            assert res is None or not res.fired, f"{q.id} misfired: {q.question}"
