"""Relation-resolver edge cases (no model load; runs in well under a second).

The resolver's output is treated as authoritative -- retrieval pins the targets'
records above everything else -- so these tests pin both halves of its contract:
it resolves the phrasings people actually use, and it abstains (returns None) on
any phrasing where a resolution could be confidently wrong.
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
    # synonyms, and "grand" written apart from its noun
    ("Who was the dad of David Calloway?", ["Robert Calloway"]),
    ("Who was the paternal grandpa of David Calloway?", ["Thomas Calloway"]),
    ("Who was the grand father of David Calloway?", ["Thomas Calloway"]),
    ("Who was the grand-father of David Calloway?", ["Thomas Calloway"]),
    ("Who was the grand mother of David Calloway?", ["Margaret Brandt"]),
    ("Who was the great grand father of David Calloway?", ["Henry Brandt", "Josiah Calloway"]),
    ("Who were the grand sons of Josiah Calloway?", ["James Calloway", "Robert Calloway"]),
    # chains, both forms, and mixed
    ("Who was the father of the father of David Calloway?", ["Thomas Calloway"]),
    ("Who was David Calloway's father's father?", ["Thomas Calloway"]),
    ("Who was the father of David Calloway's father?", ["Thomas Calloway"]),
    # "great" counts generations; paternal = through the father, so both of his
    # grandfathers are paternal great-grandfathers
    ("Who was the great-grandfather of David Calloway?", ["Henry Brandt", "Josiah Calloway"]),
    ("Who was the paternal great-grandfather of David Calloway?",
     ["Henry Brandt", "Josiah Calloway"]),
    # a possessive chain may end in ordinary words about its last person
    ("When was David Calloway's father born?", ["Robert Calloway"]),
    # a following clause doesn't disturb the parse, and "Nottingham" isn't "not"
    ("Who was the paternal grandfather of David Calloway, born near Nottingham?",
     ["Thomas Calloway"]),
    # entity linking: a doubled letter typed once, and a first name unique in the graph
    ("Who was the paternal grandfather of David Caloway?", ["Thomas Calloway"]),
    ("Who was the paternal grandfather of David?", ["Thomas Calloway"]),
    ("Who was the father of David in 1910?", ["Robert Calloway"]),
    ("Who was David's father?", ["Robert Calloway"]),
    # a fuzzy anchor carries its possessive chain: the longest linked span wins
    ("Who was the mother of David Caloway's father?", ["Margaret Brandt"]),
    # descendants
    ("Who were the grandchildren of Josiah Calloway?",
     ["Alice Calloway", "Clara Mercer", "James Calloway", "Robert Calloway"]),
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


def test_missing_generation_finds_nobody(resolver):
    # the tree has no great-great-grandparents: an empty answer, not the nearest generation
    res = resolver.resolve("Who was the great-great-grandfather of David Calloway?")
    assert res is not None and not res.fired


# --------------------------- abstains (returns None) -------------------------- #
@pytest.mark.parametrize("query", [
    # negation and exclusion, with and without the apostrophe
    "Who was not the paternal grandfather of David Calloway?",
    "Who wasn't the paternal grandfather of David Calloway?",
    "Who wasnt the grandfather of David Calloway?",
    "Who cannot be the father of David Calloway?",
    "Who else was a grandfather of David Calloway?",
    "Who was the other grandfather of Ruth Calloway?",
    "Which grandfather of David Calloway, other than Thomas, is recorded?",
    # relations the graph doesn't model
    "Who was the uncle of David Calloway?",
    "Who was David Calloway's uncle?",
    "Who was David Calloway's maternal uncle?",
    "Who was David Calloway's father-in-law?",
    "Who was the father-in-law of David Calloway?",
    "Who was David Calloway's cousin once removed?",
    "Who was David Calloway's cousin twice removed?",
    # a qualifier on the kin term changes who is meant
    "Who was the half brother of Robert Calloway?",
    "Who was the ex-husband of Susan Calloway?",
    "Who was the late husband of Susan Calloway?",
    "Who was the ex-wife of Frank Mercer?",
    "Who was the eldest son of Thomas Calloway?",
    "Who was the last child of Thomas Calloway?",
    "Who was the only daughter of Thomas Calloway?",
    "Who was the father of the first wife of Robert Calloway?",
    # a possessive chain that leaves kinship, or hides an appositive
    "Who was David Calloway's father's eldest brother?",
    "Who was David Calloway's father's ex-wife?",
    "Who was David Calloway's father's best friend?",
    "Who was Thomas Calloway's son Robert's wife?",
    "What did David Calloway's 1910 census entry record?",
    # coordination: two relations, or two people
    "Who were David Calloway's father and mother?",
    "Who were the father and mother of David Calloway?",
    "Who were David Calloway's father and uncle?",
    "Who were the aunt and mother of David Calloway?",
    "Who were the parents of Robert Calloway and Wilhelm Brandt?",
    "Who was the father of Susan Calloway and the mother of Eric Brandt?",
    "Who was the father of David Calloway and who was the mother of David Calloway?",
    "Who were the father of David Calloway and the mother of Ruth Calloway?",
    # modifiers the relation can't take, or that contradict each other
    "Who was the maternal father of David Calloway?",
    "Who was the great brother of Robert Calloway?",
    "Who was the maternal paternal grandfather of David Calloway?",
    # names that aren't anyone in the graph, however close
    "Who was the father of Thomas Jefferson?",
    "Who were the children of Charles Darwin?",
    "Who was the father of Grace Kelly?",
    "Who was the mother of Robert E. Lee?",
    "Who was the father of Walter Brandt?",
    "Who was the husband of Susan Smith?",
    "Who was the father of Ruth Brandt?",
    "Who was the father of Eric Brandt?",  # one substitution from Erik Brandt
    "Who was the father of Walker Calloway?",  # one substitution from Walter Calloway
    "Who was the father of Thomas Calloway Jr?",
    # the original non-kinship guards
    "Which children were enumerated in the household of Thomas Calloway in the 1910 census?",
    "Which source record documents the marriage of Thomas Calloway and Margaret Brandt?",
    "In what year was Thomas Calloway born?",
])
def test_abstains(resolver, query):
    assert resolver.resolve(query) is None


def test_query_length_cap(graph, resolver):
    base = "Who was the paternal grandfather of David Calloway?"
    padded = base + " " * (MAX_QUERY_CHARS - len(base))  # exactly at the cap: still parsed
    assert names(graph, resolver.resolve(padded)) == ["Thomas Calloway"]
    assert resolver.resolve(padded + " ") is None  # one past the cap: abstains


# --------------------------- ambiguous anchors -------------------------------- #
def _tiny_graph(tmp_path: Path) -> NetworkXGraphStore:
    """Two Johns; and two Lisas whose surnames are each one safe typo from 'Mattson'."""
    persons = [
        {"id": "A", "name": "John Harrow", "sex": "M"},
        {"id": "B", "name": "John Barrow", "sex": "M"},
        {"id": "D", "name": "Peter Harrow", "sex": "M"},
        {"id": "L1", "name": "Lisa Matson", "sex": "F"},
        {"id": "L2", "name": "Lisa Mattsson", "sex": "F"},
        {"id": "K", "name": "Karl Matson", "sex": "M"},
    ]
    rels = [
        {"src": "D", "dst": "A", "rel": "CHILD_OF"},
        {"src": "A", "dst": "D", "rel": "PARENT_OF"},
        {"src": "K", "dst": "L1", "rel": "CHILD_OF"},
        {"src": "L1", "dst": "K", "rel": "PARENT_OF"},
    ]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"persons": persons, "relationships": rels, "person_documents": {}}))
    return NetworkXGraphStore().load(path)


def test_ambiguous_first_name_abstains(tmp_path):
    assert RelationResolver(_tiny_graph(tmp_path)).resolve("Who was the son of John?") is None


def test_substitution_typo_never_links(tmp_path):
    # "darrow" is one substitution from both "harrow" and "barrow" -- and from neither safely
    r = RelationResolver(_tiny_graph(tmp_path))
    assert r.resolve("Who was the son of John Darrow?") is None


def test_safe_typo_ambiguous_abstains(tmp_path):
    # "mattson" is a doubled letter away from both "matson" and "mattsson"
    r = RelationResolver(_tiny_graph(tmp_path))
    assert r.resolve("Who was the son of Lisa Mattson?") is None


def test_unique_safe_typo_links(tmp_path):
    res = RelationResolver(_tiny_graph(tmp_path)).resolve("Who was the son of John Harow?")
    assert res is not None and res.fired and res.targets == ["D"] and res.linked_by == "fuzzy"


# --------------------------- gold set unchanged ------------------------------ #
def test_gold_relational_questions_resolve_to_their_answer(resolver):
    for q in load_questions():
        res = resolver.resolve(q.question)
        if q.category == "relational":
            assert res is not None and res.fired, q.id
            assert q.answer_entity in res.targets, q.id
        else:
            assert res is None or not res.fired, f"{q.id} misfired: {q.question}"
