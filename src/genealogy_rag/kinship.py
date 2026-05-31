"""Structured kinship resolution -- the KG-augmented-QA layer.

A genealogical question like "who was the maternal grandfather of Nancy Ainsworth?"
never names the answer entity, so lexical/dense retrieval cannot surface the
answer's records. This module parses the *relation* and *anchor* from the query
and resolves the target person(s) by traversing the typed kinship graph, so the
retriever can then pull the target's source documents.

It fires only on the pattern `<relation-headword> of <PersonName>` -- so
"grandfather of X" resolves, but "household of X" or "marriage of X and Y" do not
(their headword is not a kin term). When it resolves nothing, the retriever falls
back to fuzzy retrieval; when it resolves, the result is treated as authoritative.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .graph.base import GraphStore

RELATION_HEADWORDS = {
    "grandfather", "grandmother", "grandparent", "grandparents",
    "father", "mother", "parent", "parents",
    "son", "sons", "daughter", "daughters", "child", "children",
    "husband", "wife", "spouse",
    "brother", "brothers", "sister", "sisters", "sibling", "siblings",
    "cousin", "cousins",
}
MODIFIERS = {"maternal", "paternal", "great"}
_WORD = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Resolution:
    anchor: str
    relation: str
    modifiers: tuple[str, ...]
    targets: list[str]

    @property
    def fired(self) -> bool:
        return bool(self.targets)


class RelationResolver:
    def __init__(self, graph: GraphStore):
        self.g = graph
        # longest names first so "Margaret Brandt" wins over a bare "Margaret"
        self._names = sorted(graph.all_person_names().items(),
                             key=lambda kv: -len(kv[0]))

    # --- set helpers -------------------------------------------------------- #
    def _males(self, ids: list[str]) -> list[str]:
        return [i for i in ids if self.g.sex(i) == "M"]

    def _females(self, ids: list[str]) -> list[str]:
        return [i for i in ids if self.g.sex(i) == "F"]

    def _father(self, p: str) -> list[str]:
        return self._males(self.g.parents(p))

    def _mother(self, p: str) -> list[str]:
        return self._females(self.g.parents(p))

    def _siblings(self, p: str) -> list[str]:
        out: list[str] = []
        for par in self.g.parents(p):
            for c in self.g.children(par):
                if c != p and c not in out:
                    out.append(c)
        return out

    def _cousins(self, p: str) -> list[str]:
        out: list[str] = []
        for par in self.g.parents(p):
            for au in self._siblings(par):
                for c in self.g.children(au):
                    if c not in out:
                        out.append(c)
        return out

    @staticmethod
    def _dedup(xs: list[str]) -> list[str]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # --- parsing ------------------------------------------------------------ #
    def _parse(self, query: str) -> tuple[str, str, tuple[str, ...]] | None:
        ql = query.lower()
        for name_lower, pid in self._names:
            marker = f"of {name_lower}"
            j = ql.find(marker)
            if j == -1:
                continue
            prefix_words = _WORD.findall(ql[:j])
            if not prefix_words:
                continue
            headword = prefix_words[-1]  # the word that *governs* "of <name>"
            if headword not in RELATION_HEADWORDS:
                continue  # "household of X", "marriage of X" -> not a kin lookup
            mods = tuple(w for w in prefix_words[-4:-1] if w in MODIFIERS)
            return pid, headword, mods
        return None

    # --- resolution --------------------------------------------------------- #
    def resolve(self, query: str) -> Resolution | None:
        parsed = self._parse(query)
        if parsed is None:
            return None
        anchor, rel, mods = parsed
        maternal, paternal, great = (
            "maternal" in mods, "paternal" in mods, "great" in mods)

        def grandparents() -> list[str]:
            if paternal:
                sources = self._father(anchor)
            elif maternal:
                sources = self._mother(anchor)
            else:
                sources = self.g.parents(anchor)
            gps = [gp for s in sources for gp in self.g.parents(s)]
            if great:  # one further generation along the same frontier
                gps = [x for g in gps for x in self.g.parents(g)]
            return self._dedup(gps)

        if rel in ("father",):
            targets = self._father(anchor)
        elif rel in ("mother",):
            targets = self._mother(anchor)
        elif rel in ("parent", "parents"):
            targets = self.g.parents(anchor)
        elif rel in ("grandfather",):
            targets = self._males(grandparents())
        elif rel in ("grandmother",):
            targets = self._females(grandparents())
        elif rel in ("grandparent", "grandparents"):
            targets = grandparents()
        elif rel in ("son", "sons"):
            targets = self._males(self.g.children(anchor))
        elif rel in ("daughter", "daughters"):
            targets = self._females(self.g.children(anchor))
        elif rel in ("child", "children"):
            targets = self.g.children(anchor)
        elif rel in ("husband",):
            targets = self._males(self.g.spouses(anchor))
        elif rel in ("wife",):
            targets = self._females(self.g.spouses(anchor))
        elif rel in ("spouse",):
            targets = self.g.spouses(anchor)
        elif rel in ("brother", "brothers"):
            targets = self._males(self._siblings(anchor))
        elif rel in ("sister", "sisters"):
            targets = self._females(self._siblings(anchor))
        elif rel in ("sibling", "siblings"):
            targets = self._siblings(anchor)
        elif rel in ("cousin", "cousins"):
            targets = self._cousins(anchor)
        else:  # pragma: no cover
            targets = []

        return Resolution(anchor=anchor, relation=rel, modifiers=mods,
                          targets=self._dedup(targets))
