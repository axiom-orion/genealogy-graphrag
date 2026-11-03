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

Synonyms map to canonical terms (dad -> father, grandpa -> grandfather), "grand
father" is read as "grandfather", and "great" may repeat (great-great-grandfather
is one generation past great-grandfather). "household of X" or "marriage of X and
Y" do not fire: their headword is not a kin term.

When it resolves, the result is treated as AUTHORITATIVE -- the retriever pins the
targets' records above fuzzy retrieval -- so a wrong resolution is worse than none.
The resolver therefore ABSTAINS (returns None, and retrieval falls back to the fused
dense+lexical ranking) whenever the question is not plainly one relation path from
one person:

  * negation or exclusion        "who was NOT ...", "the OTHER grandfather", "who else"
  * a relation it doesn't model   uncle, aunt, in-laws, step-/half-kin, "once removed"
  * any qualifier on a kin term   "eldest son", "only daughter", "ex-husband", "late wife"
  * coordination                  "father and mother of X", "parents of X and Y"
  * a modifier the relation can't take, or contradictory ones   "maternal father"
  * anything but exactly one person mentioned, or a name it can't link to one person
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
SIDES = {"maternal", "paternal"}
GREAT = "great"
DETERMINERS = {"the", "a", "an", "his", "her", "their", "its"}
# The only words allowed right before a kin term (and its modifiers) in the
# "<relation> of <person>" form. Anything else is a qualifier the graph can't honour
# ("eldest son", "ex husband", "only daughter", "late wife") or a coordination
# ("father and mother") -- both abstain.
BEFORE_HEAD_OK = DETERMINERS | {
    "who", "whom", "whose", "which", "what", "was", "were", "is", "are", "be", "been",
    "list", "name", "find", "identify", "show", "give", "tell", "me", "us", "about",
}
# A bare first name links only when what follows it cannot be a surname.
FOLLOWS_FIRST_NAME_OK = {
    "in", "who", "whose", "when", "where", "born", "was", "is", "were", "died", "during",
    "on", "at", "from", "before", "after", "lived", "married", "by", "around", "circa",
}
NAME_SUFFIXES = {"jr", "sr", "junior", "senior", "ii", "iii", "iv"}
# The only words allowed right before an anchor name. Anything else may be part of a
# longer name ("Mary Susan Calloway", "John Thomas Calloway") or a coordination. Titles
# abstain too: in old records "Mrs. Thomas Calloway" names Thomas's WIFE.
BEFORE_ANCHOR_OK = DETERMINERS | {
    "of", "who", "whom", "whose", "which", "what", "when", "where", "was", "were", "is",
    "are", "did", "does", "do", "about", "for", "me", "tell", "list", "name", "find",
    "show", "give",
}
# In the "<relation> of <person>" form, the words after the name must open with one of
# these. Anything else may continue the name ("Calloway Smith", "Calloway the younger")
# or add another person ("& Ruth", "plus Ruth", "with Walter").
AFTER_ANCHOR_OK = {
    "in", "born", "who", "whose", "when", "where", "during", "at", "around", "circa",
    "living", "lived", "died", "married", "before", "after",
}
# Words that, anywhere after the relation, change which line or relation is meant
# ("through his mother", "on his mother's side", "once removed").
LINEAGE_WORDS = (
    SIDES | {"side", "sides", "line", "lineage", "branch", "removed", "once", "twice"}
    | set(RELATIONS) | set(SYNONYMS) | UNSUPPORTED_KIN
)
# The first word after a possessive chain's last kin term may not start a qualifier
# or another person.
AFTER_CHAIN_BLOCK = {
    "and", "or", "nor", "plus", "with", "as", "through", "via", "on", "from", "by",
    "but", "minus", "once", "twice", "thrice", "removed", "in", "law",
}

_WORD = re.compile(r"[a-z]+")
_NEGATION = re.compile(
    r"\b(?:not|never|nor|no|none|neither|cannot|without|but|minus|except|exclude|excluded|"
    r"excluding|omit|omitting|ignore|ignoring|besides|else|other|another|"
    r"other\s+than|apart\s+from|aside\s+from|rather\s+than|instead\s+of|"
    r"isnt|wasnt|arent|werent|dont|doesnt|didnt|cant|couldnt|wouldnt|shouldnt|"
    r"hasnt|havent|hadnt)\b|n't\b"
)
_OF_BEFORE = re.compile(r"\bof\s+$")
_OF_THEN_SPACE = re.compile(r"\bof\s+")
_OF = re.compile(r"\bof\b")
# "grand father", "grand-father", "great grand son" -> one word, so "grand" is never
# silently dropped (which would answer one generation short).
_SPLIT_GRAND = re.compile(
    r"\bgrand[\s-]+(?=(?:father|mother|parent|son|daughter|child|dad|kid)s?\b|children\b)"
)
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'"})

# Fuzzy entity linking (only when no full name appears verbatim): a 2-3 word span
# links to a full name only when it is one SAFE typo away -- a doubled letter typed
# once or twice, or two adjacent letters swapped -- and to exactly one person. A
# letter substitution ("Eric" for "Erik", "Walker" for "Walter") may be a different
# person and does not link. A 1-word span links on a first name only when unique.
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


@dataclass(frozen=True)
class _Mention:
    start: int
    end: int
    text: str
    pid: str
    linked_by: Linking


def _is_safe_typo(typed: str, name: str) -> bool:
    """One adjacent swap, or one doubled letter typed once / a letter typed twice."""
    if typed == name:
        return False
    if len(typed) == len(name):
        diff = [i for i, (a, b) in enumerate(zip(typed, name, strict=True)) if a != b]
        return (len(diff) == 2 and diff[1] == diff[0] + 1
                and typed[diff[0]] == name[diff[1]] and typed[diff[1]] == name[diff[0]])
    if abs(len(typed) - len(name)) != 1:
        return False
    longer, shorter = (typed, name) if len(typed) > len(name) else (name, typed)
    for i, ch in enumerate(longer):
        if longer[:i] + longer[i + 1:] == shorter:
            # the extra letter must double a neighbour: "Calloway" vs "Caloway"
            return (i > 0 and longer[i - 1] == ch) or (i + 1 < len(longer) and longer[i + 1] == ch)
    return False


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
    def _is_modifier(word: str) -> bool:
        return word in SIDES or word == GREAT

    def _of_steps(self, prefix: str) -> list[Step] | None:
        """Steps from `... <rel> of [the] <rel> of` (the text before the anchor's
        `of`), anchor-outward. Every segment must END in a kin term preceded only by
        modifiers and an allowed word; inner segments may hold nothing else."""
        segments = _OF.split(prefix)
        steps: list[Step] = []
        for k, segment in enumerate(segments):
            words = _WORD.findall(segment)
            if not words or words[-1] in UNSUPPORTED_KIN:
                return None
            head_at = len(words) - 1
            i = head_at
            while i > 0 and self._is_modifier(words[i - 1]):
                i -= 1
            if i > 0 and words[i - 1] not in BEFORE_HEAD_OK:
                return None  # "eldest son", "ex husband", "father and mother", ...
            if k > 0 and any(w not in DETERMINERS for w in words[:i]):
                return None  # inner hop with extra words: "father of the first wife of X"
            step = self._step(words[i:head_at], words[head_at])
            if step is None:
                return None
            steps.append(step)
        return list(reversed(steps))  # written outer-first; applied anchor-outward

    def _possessive_steps(self, rest: str) -> list[Step] | None:
        """Steps from `'s <rel>'s <rel> ...` after the anchor, anchor-outward. Every
        segment must START with a kin term (after modifiers); only the last may carry
        trailing words ("X's father born"), and never a coordination or qualifier."""
        segments = rest.split("'s")
        steps: list[Step] = []
        for idx, segment in enumerate(segments):
            words = _WORD.findall(segment)
            last = idx == len(segments) - 1
            if not words:
                if last and steps:
                    break  # a trailing "'s" with nothing after it
                return None
            head_at = 0
            while head_at < len(words) and self._is_modifier(words[head_at]):
                head_at += 1
            if head_at == len(words):
                return None  # modifiers with no head: "X's great"
            step = self._step(words[:head_at], words[head_at])
            if step is None:
                return None  # "X's eldest son", "X's uncle", "X's father's friend"
            trailing = words[head_at + 1:]
            if trailing and not last:
                return None  # an appositive inside the chain: "X's son Robert's wife"
            if trailing and (trailing[0] in AFTER_CHAIN_BLOCK
                             or any(w in LINEAGE_WORDS for w in trailing)):
                return None  # "X's father and mother", "X's grandfather through his mother"
            steps.append(step)
        return steps or None

    # --- parsing: entity linking ------------------------------------------- #
    @staticmethod
    def _possessive_tail(q: str, end: int, text: str) -> str | None:
        """The text after the possessive marker following a mention, or None."""
        if q.startswith("'s", end):
            return q[end + 2:]
        if q.startswith("'", end) and text.endswith("s"):
            return q[end + 1:]
        return None

    @staticmethod
    def _word_after(q: str, end: int) -> str | None:
        m = _WORD.search(q, end)
        return m.group(0) if m else None

    @staticmethod
    def _word_before(q: str, start: int) -> str | None:
        words = _WORD.findall(q[max(0, start - 40):start])
        return words[-1] if words else None

    def _exact_mentions(self, q: str) -> list[tuple[int, int, str, str]]:
        """Every verbatim full-name mention, longest names first, no overlaps."""
        covered = bytearray(len(q))
        found: list[tuple[int, int, str, str]] = []
        for pattern, name, pid in self._name_patterns:
            for m in pattern.finditer(q):
                s, e = m.span()
                if any(covered[s:e]):
                    continue  # inside a longer name already matched
                covered[s:e] = b"\x01" * (e - s)
                found.append((s, e, name, pid))
        return found

    def _fuzzy_mentions(self, q: str, in_position) -> list[_Mention]:
        """Typo / first-name links for spans in a relation position, longest first."""
        found: list[_Mention] = []
        words = list(_WORD.finditer(q))
        for i in range(len(words)):
            for n in range(1, MAX_ANCHOR_SPAN_WORDS + 1):
                if i + n > len(words):
                    break
                s, e = words[i].start(), words[i + n - 1].end()
                text = q[s:e]
                if not in_position(s, e, text):
                    continue
                if n == 1:
                    pids = self._first_names.get(text, [])
                    follower = self._word_after(q, e)
                    if (len(text) >= MIN_FIRST_NAME_LENGTH and len(pids) == 1
                            and (follower is None or follower in FOLLOWS_FIRST_NAME_OK
                                 or self._possessive_tail(q, e, text) is not None)):
                        found.append(_Mention(s, e, text, pids[0], "first_name"))
                elif len(text) >= MIN_FUZZY_NAME_LENGTH:
                    close = [pid for name, pid in self._names if _is_safe_typo(text, name)]
                    if len(close) == 1:
                        found.append(_Mention(s, e, text, close[0], "fuzzy"))
        # Longest span first: in "mother of David Caloway's father" the anchor mention
        # is "david caloway" (whose possessive carries the chain), not the bare "david".
        found.sort(key=lambda m: m.start - m.end)
        return found

    def _anchor(self, q: str) -> _Mention | None:
        """The one person the relation path hangs off, or None (abstain)."""
        # Offsets that directly follow an "of " -- computed once, so checking a span's
        # position is O(1) rather than a rescan of everything before it.
        after_of = {m.end() for m in _OF_THEN_SPACE.finditer(q)}

        def in_position(start: int, end: int, text: str) -> bool:
            return start in after_of or self._possessive_tail(q, end, text) is not None

        exact = self._exact_mentions(q)
        if exact:
            if len(exact) != 1:
                return None  # two people, or one person twice: not one relation path
            s, e, name, pid = exact[0]
            mention = _Mention(s, e, name, pid, "exact") if in_position(s, e, name) else None
        else:
            fuzzy = self._fuzzy_mentions(q, in_position)
            if len({m.pid for m in fuzzy}) != 1:
                return None  # nobody linked, or the candidate spans disagree
            mention = fuzzy[0]
        if mention is None:
            return None
        # "Mary Susan Calloway" and "Thomas Calloway Jr" are someone else; "Ruth and X"
        # asks about two people. (After a possessive the next word is the "s" of "'s".)
        before = self._word_before(q, mention.start)
        if before is not None and before not in BEFORE_ANCHOR_OK:
            return None
        if self._word_after(q, mention.end) in NAME_SUFFIXES:
            return None
        return mention

    def _parse(self, query: str) -> tuple[str, list[Step], Linking] | None:
        if len(query) > MAX_QUERY_CHARS:
            return None
        q = _SPLIT_GRAND.sub("grand", query.lower().translate(_APOSTROPHES))
        if _NEGATION.search(q):
            return None
        anchor = self._anchor(q)
        if anchor is None:
            return None
        steps: list[Step] = []
        tail = self._possessive_tail(q, anchor.end, anchor.text)
        if tail is not None:
            poss = self._possessive_steps(tail)
            if poss is None:
                return None
            steps += poss
        of_before = _OF_BEFORE.search(q[: anchor.start])
        if of_before is not None and tail is None:
            # "<relation> of <person> ...": what follows the name may not continue it,
            # add a person, or qualify the relation ("once removed", "through his mother").
            after = _WORD.findall(q[anchor.end:])
            if after and (after[0] not in AFTER_ANCHOR_OK
                          or any(w in LINEAGE_WORDS for w in after)):
                return None
        if of_before is not None:
            outer = self._of_steps(q[: of_before.start()])
            if outer is None:
                return None
            steps += outer
        if not steps:
            return None
        return anchor.pid, steps, anchor.linked_by

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
