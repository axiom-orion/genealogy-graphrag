"""Provenance assembly. Every answer carries the bibliographic citations of the
documents it rests on -- so a claim is always traceable to a named source.

Note: 'provenance' here means *source citation* (the genealogical sense), not a
cryptographic proof-chain. Deliberately a generic, textbook concept."""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import Document


@dataclass(frozen=True)
class Citation:
    doc_id: str
    title: str
    citation: str
    doc_type: str


@dataclass(frozen=True)
class GroundedAnswer:
    query: str
    context: list[Document]
    citations: list[Citation]

    def render(self) -> str:
        lines = [f"Q: {self.query}", "", "Top supporting sources:"]
        for i, c in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {c.title} ({c.doc_type})")
            lines.append(f"      {c.citation}")
        return "\n".join(lines)


def build_citations(docs: list[Document]) -> list[Citation]:
    return [Citation(doc_id=d.id, title=d.title, citation=d.citation,
                     doc_type=d.type) for d in docs]
