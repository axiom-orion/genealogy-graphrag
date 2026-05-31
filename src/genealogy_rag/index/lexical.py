"""Sparse lexical index (BM25). Complements dense retrieval on rare proper nouns
(surnames, place names) where embeddings under-perform."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LexicalIndex:
    def __init__(self):
        self._bm25: BM25Okapi | None = None

    def build(self, texts: list[str]) -> LexicalIndex:
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])
        return self

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
