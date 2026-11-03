"""Structured kinship resolution -- the KG-augmented-QA layer.

A genealogical question like "who was the maternal grandfather of Nancy Ainsworth?"
never names the answer entity, so lexical/dense retrieval cannot surface the
answer's records. This module parses the *relation path* and *anchor* from the
query and resolves the target person(s) by traversing the typed kinship graph, so
the retriever can then pull the target's source documents.

It fires on a relation path anchored on one person, written either way, and on
chains of either:

    "<relation> of <Person>"   "paternal grandfather of David Calloway"
    "<Person>'s <relation>"    "David Calloway's paternal grandfather"
    chains                     "father of the father of X", "X's mother's father",
                               "grandfather of X's sister"

Synonyms map to canonical terms (dad -> father, grandpa -> grandfather) and
"great" may repeat (great-great-grandfather is one generation past
great-grandfather). "household of X" or "marriage of X and Y" do not fire: their
headword is not a kin term.

When it resolves, the result is treated as AUTHORITATIVE -- the retriever pins the
targets' records above fuzzy retrieval -- so a wrong resolution is worse than none.
The resolver therefore ABSTAINS (returns None, and retrieval falls back to the fused
dense+lexical ranking) whenever it is not sure:

  * negation or exclusion        "who was NOT the grandfather of X", "other than"
  * a relation it doesn't model   uncle, aunt, nephew, in-laws, step-/half-kin
  * a modifier the relation can't take   "maternal father", "great brother"
  * contradictory modifiers       "maternal paternal grandfather"
  * an anchor it can't link to exactly one person, or more than one anchor
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .graph.base import GraphStore

# --- vocabulary ------------------------------------------------------------- #
Direction = Literal["up", "down", "spouse", "sibling", "cousin"]


@dataclass(frozen=True)
class _RelationSpec:
    direction: Direction
    generations: int = 0  # for up/down: 1 = parent/child, 2 = grandparent/grandchild
    sex: str | None = None  # "M" / "F" filter on the final targets, or None
    takes_side: bool = False  # accepts maternal/paternal
    takes_great: bool = False  # accepts great-, great-great-, ...


RELATIONS: dict[str, _RelationSpec] = {
    "father": _RelationSpec("up", 1, "M"),
    "mother": _RelationSpec("up", 1, "F"),
    "parent": _RelationSpec("up", 1),
    "grandfather": _RelationSpec("up", 2, "M", takes_side=True, takes_great=True),
    "grandmother": _RelationSpec("up", 2, "F", takes_side=True, takes_great=True),
    "grandparent": _RelationSpec("up", 2, takes_side=True, takes_great=True),
    "son": _RelationSpec("down", 1, "M"),
    "daughter": _RelationSpec("down", 1, "F"),
    "child": _RelationSpec("down", 1),
    "grandson": _RelationSpec("down", 2, "M", takes_great=True),
    "granddaughter": _RelationSpec("down", 2, "F", takes_great=True),
    "grandchild": _RelationSpec("down", 2, takes_great=True),
    "husband": _RelationSpec("spouse", sex="M"),
    "wife": _RelationSpec("spouse", sex="F"),
    "spouse": _RelationSpec("spouse"),
    "brother": _RelationSpec("sibling", sex="M"),
    "sister": _RelationSpec("sibling", sex="F"),
    "sibling": _RelationSpec("sibling"),
    "cousin": _RelationSpec("cousin"),
}

# Plurals and everyday synonyms -> canonical relation term.
SYNONYMS: dict[str, str] = {
    "fathers": "father", "dad": "father", "daddy": "father", "papa": "father",
    "mothers": "mother", "mom": "mother", "mum": "mother", "mama": "mother",
    "mommy": "mother", "mummy": "mother",
    "parents": "parent",
    "grandfathers": "grandfather", "grandpa": "grandfather", "granddad": "grandfather",
    "grandad": "grandfather", "granddaddy": "grandfather",
    "grandmothers": "grandmother", "grandma": "grandmother", "granny": "grandmother",
    "grandmama": "grandmother",
    "grandparents": "grandparent",
    "sons": "son", "daughters": "daughter", "children": "child", "kids": "child",
    "grandsons": "grandson", "granddaughters": "granddaughter",
    "grandchildren": "grandchild", "grandkids": "grandchild",
    "husbands": "husband", "wives": "wife", "spouses": "spouse",
    "brothers": "brother", "sisters": "sister", "siblings": "sibling",
    "cousins": "cousin",
}

# Kin terms the graph cannot answer. Hitting one means "the question is about a
# relation we don't model" -- abstain rather than answer a nearby relation.
UNSUPPORTED_KIN = {
    "uncle", "uncles", "aunt", "aunts", "auntie", "nephew", "nephews", "niece", "nieces",
    "stepfather", "stepmother", "stepson", "stepdaughter", "stepchild", "stepchildren",
    "stepbrother", "stepsister", "godfather", "godmother", "godson", "goddaughter",
    "godparent", "godchild", "ancestor", "ancestors", "descendant", "descendants",
    "relative", "relatives", "kin", "inlaw", "inlaws",
}
# A qualifier right before a kin term changes its meaning ("half brother",
# "step father", "foster mother"): the graph's PARENT_OF edges can't tell.
ABSTAIN_QUALIFIERS = {
    "half", "step", "foster", "adoptive", "adopted", "god", "birth", "former", "first", "second",
}
SIDES = {"maternal", "paternal"}
GREAT = "great"
DETERMINERS = {"the", "a", "an", "his", "her", "their", "its"}

_WORD = re.compile(r"[a-z]+")
_NEGATION = re.compile(
    r"\b(?:not|never|nor|except|excluding|besides|other\s+than|apart\s+from|"
    r"aside\s+from|rather\s+than|instead\s+of)\b|n't\b"
)
_OF_BEFORE = re.compile(r"\bof\s+$")
_OF_THEN_SPACE = re.compile(r"\bof\s+")
_OF = re.compile(r"\bof\b")
_IN_LAW = re.compile(r"^\W*in\W+laws?\b")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})

# Fuzzy entity linking: a 2-3 word span within this edit distance of exactly one
# full name links to it; a 1-word span links on a first name only when unique.
MAX_NAME_EDIT_DISTANCE = 1
MIN_FUZZY_NAME_LENGTH = 8
MIN_FIRST_NAME_LENGTH = 3
MAX_ANCHOR_SPAN_WORDS = 3
# A question is a sentence, not a document: longer input abstains rather than being
# parsed (and keeps the resolver's cost bounded on pathological input).
MAX_QUERY_CHARS = 1_000

Linking = Literal["exact", "fuzzy", "first_name"]


@dataclass(frozen=True)
class Step:
    """One hop of a relation path: a canonical relation plus its modifiers."""

    relation: str
    side: str | None = None  # "maternal" / "paternal"
    greats: int = 0

    @property
    def modifiers(self) -> tuple[str, ...]:
        return ((self.side,) if self.side else ()) + (GREAT,) * self.greats


@dataclass(frozen=True)
class Resolution:
    anchor: str
    relation: str  # the outermost relation (what the question asks for)
    modifiers: tuple[str, ...]  # the outermost relation's modifiers
    targets: list[str]
    path: tuple[Step, ...] = field(default=())  # anchor-outward hops
    linked_by: Linking = "exact"

    @property
    def fired(self) -> bool:
        return bool(self.targets)


def _edit_distance(a: str, b: str, cap: int) -> int:
    """Levenshtein distance, short-circuiting once every cell exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


class RelationResolver:
    def __init__(self, graph: GraphStore):
        self.g = graph
        # longest names first so "Margaret Brandt" wins over a bare "Margaret"
        self._names = sorted(graph.all_person_names().items(),
                             key=lambda kv: -len(kv[0]))
        self._name_patterns = [
            (re.compile(rf"(?<![a-z]){re.escape(name)}(?![a-z])"), name, pid)
            for name, pid in self._names
        ]
        firsts: dict[str, list[str]] = {}
        for name, pid in self._names:
            firsts.setdefault(name.split()[0], []).append(pid)
        self._first_names = firsts

    # --- set helpers -------------------------------------------------------- #
    def _males(self, ids: list[str]) -> list[str]:
        return [i for i in ids if self.g.sex(i) == "M"]

    def _females(self, ids: list[str]) -> list[str]:
        return [i for i in ids if self.g.sex(i) == "F"]

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

    def _apply(self, step: Step, people: list[str]) -> list[str]:
        """Everyone `step` away from anyone in `people`."""
        spec = RELATIONS[step.relation]
        out: list[str] = []
        for p in people:
            if spec.direction in ("up", "down"):
                hop = self.g.parents if spec.direction == "up" else self.g.children
                frontier = hop(p)
                # the side picks the line on the first hop: paternal = via the father
                if step.side == "paternal":
                    frontier = self._males(frontier)
                elif step.side == "maternal":
                    frontier = self._females(frontier)
                for _ in range(spec.generations + step.greats - 1):
                    frontier = [x for f in frontier for x in hop(f)]
            elif spec.direction == "spouse":
                frontier = self.g.spouses(p)
            elif spec.direction == "sibling":
                frontier = self._siblings(p)
            else:
                frontier = self._cousins(p)
            if spec.sex == "M":
                frontier = self._males(frontier)
            elif spec.sex == "F":
                frontier = self._females(frontier)
            out.extend(frontier)
        return self._dedup(out)

    # --- parsing: one hop --------------------------------------------------- #
    @staticmethod
    def _step(mods: list[str], head: str) -> Step | None:
        """Validate `mods head` as one hop; None when it must abstain."""
        relation = SYNONYMS.get(head, head)
        spec = RELATIONS.get(relation)
        if spec is None:
            return None
        sides = [m for m in mods if m in SIDES]
        greats = mods.count(GREAT)
        if len(set(sides)) > 1 or (sides and not spec.takes_side):
            return None
        if greats and not spec.takes_great:
            return None
        return Step(relation, sides[0] if sides else None, greats)

    @staticmethod
    def _split_head(words: list[str], head_at: int) -> tuple[list[str], str] | None:
        """The head word at `head_at` plus the run of modifiers right before it."""
        i = head_at
        while i > 0 and (words[i - 1] in SIDES or words[i - 1] == GREAT):
            i -= 1
        if i > 0 and words[i - 1] in ABSTAIN_QUALIFIERS:
            return None
        return words[i:head_at], words[head_at]

    def _of_steps(self, prefix: str) -> list[Step] | None:
        """Steps from `... <rel> of [the] <rel> of` (the text before the anchor's
        `of`), anchor-outward. Every segment must END in a kin term; inner segments
        may hold nothing but a determiner, modifiers and that term."""
        segments = _OF.split(prefix)
        steps: list[Step] = []
        for k, segment in enumerate(segments):
            words = _WORD.findall(segment)
            if not words:
                return None
            if words[-1] in UNSUPPORTED_KIN:
                return None
            split = self._split_head(words, len(words) - 1)
            if split is None:
                return None
            mods, head = split
            leading = words[: len(words) - 1 - len(mods)]
            if k > 0 and any(w not in DETERMINERS for w in leading):
                return None  # "father of the first wife of X": an unmodelled qualifier
            step = self._step(mods, head)
            if step is None:
                return None
            steps.append(step)
        return list(reversed(steps))  # written outer-first; applied anchor-outward

    def _possessive_steps(self, rest: str) -> list[Step] | None:
        """Steps from `'s <rel>'s <rel> ...` after the anchor, anchor-outward. The
        chain ends at the first segment that is not a kin term ("X's father's farm");
        an unmodelled kin term or an in-law suffix abstains instead."""
        steps: list[Step] = []
        for segment in rest.split("'s"):
            words = _WORD.findall(segment)
            if not words:
                break
            head_at = 0
            while head_at < len(words) and (words[head_at] in SIDES or words[head_at] == GREAT):
                head_at += 1
            if head_at == len(words):
                return None  # modifiers with no head: "X's great"
            head = words[head_at]
            if head in UNSUPPORTED_KIN or head in ABSTAIN_QUALIFIERS:
                return None
            relation = SYNONYMS.get(head, head)
            if relation not in RELATIONS:
                if head_at > 0:
                    return None  # "X's maternal uncle"-style: modifiers on a non-kin word
                break
            after = segment[segment.find(head) + len(head):]
            if _IN_LAW.match(after):
                return None  # "X's father-in-law"
            step = self._step(words[:head_at], head)
            if step is None:
                return None
            steps.append(step)
        return steps or None

    # --- parsing: entity linking ------------------------------------------- #
    @staticmethod
    def _possessive_tail(q: str, end: int, name: str) -> str | None:
        """The text after the possessive marker following the anchor, or None."""
        if q.startswith("'s", end):
            return q[end + 2:]
        if q.startswith("'", end) and name.endswith("s"):
            return q[end + 1:]
        return None

    def _relation_positions(self, q: str) -> list[tuple[int, int, str, str, Linking]]:
        """Anchor mentions in a relation position (right after `of`, or right before
        a possessive), as (start, end, matched text, person id, how it linked)."""
        # Offsets that directly follow an "of " -- computed once, so checking a span's
        # position is O(1) rather than a rescan of everything before it.
        after_of = {m.end() for m in _OF_THEN_SPACE.finditer(q)}

        def in_position(start: int, end: int, text: str) -> bool:
            return start in after_of or self._possessive_tail(q, end, text) is not None

        covered = bytearray(len(q))  # characters already inside a longer matched name
        exact: list[tuple[int, int, str, str, Linking]] = []
        for pattern, name, pid in self._name_patterns:
            for m in pattern.finditer(q):
                s, e = m.span()
                if any(covered[s:e]):
                    continue  # inside a longer name already matched
                covered[s:e] = b"\x01" * (e - s)
                if in_position(s, e, name):
                    exact.append((s, e, name, pid, "exact"))
        if exact:
            return exact

        # No exact name in a relation position: try fuzzy / first-name spans there.
        found: list[tuple[int, int, str, str, Linking]] = []
        words = list(re.finditer(r"[a-z]+", q))
        for i in range(len(words)):
            for n in range(1, MAX_ANCHOR_SPAN_WORDS + 1):
                if i + n > len(words):
                    break
                s, e = words[i].start(), words[i + n - 1].end()
                text = q[s:e]
                if not in_position(s, e, text):
                    continue
                for pid, how in self._link_fuzzy(text, n):
                    found.append((s, e, text, pid, how))
        # Longest span first: in "mother of David Caloway's father" the anchor mention
        # is "david caloway" (whose possessive carries the chain), not the bare "david".
        found.sort(key=lambda m: m[0] - m[1])
        return found

    def _link_fuzzy(self, span: str, n_words: int) -> list[tuple[str, Linking]]:
        if n_words == 1:
            pids = self._first_names.get(span, [])
            if len(span) >= MIN_FIRST_NAME_LENGTH and len(pids) == 1:
                return [(pids[0], "first_name")]
            return []
        if len(span) < MIN_FUZZY_NAME_LENGTH:
            return []
        close = [pid for name, pid in self._names
                 if _edit_distance(span, name, MAX_NAME_EDIT_DISTANCE) <= MAX_NAME_EDIT_DISTANCE]
        return [(close[0], "fuzzy")] if len(close) == 1 else []

    def _parse(self, query: str) -> tuple[str, list[Step], Linking] | None:
        if len(query) > MAX_QUERY_CHARS:
            return None
        q = query.lower().translate(_APOSTROPHES)
        if _NEGATION.search(q):
            return None
        mentions = self._relation_positions(q)
        if len({pid for _, _, _, pid, _ in mentions}) != 1:
            return None  # no anchor, or more than one person in a relation position
        s, e, text, pid, how = mentions[0]
        steps: list[Step] = []
        tail = self._possessive_tail(q, e, text)
        if tail is not None:
            poss = self._possessive_steps(tail)
            if poss is None:
                return None
            steps += poss
        of_before = _OF_BEFORE.search(q[:s])
        if of_before is not None:
            outer = self._of_steps(q[: of_before.start()])
            if outer is None:
                return None
            steps += outer
        if not steps:
            return None
        return pid, steps, how

    # --- resolution --------------------------------------------------------- #
    def resolve(self, query: str) -> Resolution | None:
        parsed = self._parse(query)
        if parsed is None:
            return None
        anchor, steps, how = parsed
        people = [anchor]
        for step in steps:
            people = self._apply(step, people)
        outer = steps[-1]
        return Resolution(anchor=anchor, relation=outer.relation, modifiers=outer.modifiers,
                          targets=people, path=tuple(steps), linked_by=how)
