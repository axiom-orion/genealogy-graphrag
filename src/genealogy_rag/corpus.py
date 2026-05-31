"""Corpus + gold-question loading. Swap the loader to ingest a real GEDCOM locally."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings


@dataclass(slots=True)
class Document:
    id: str
    type: str
    title: str
    text: str
    persons: list[str]
    citation: str
    facts: dict = field(default_factory=dict)

    @property
    def searchable(self) -> str:
        """Title + body is what we index; titles carry strong lexical signal."""
        return f"{self.title}. {self.text}"


@dataclass(slots=True)
class Question:
    id: str
    category: str
    question: str
    relevant_doc_ids: list[str]
    answer_entity: str | None
    answer_text: str


def load_documents(path: Path | None = None) -> list[Document]:
    path = path or settings.documents_path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_corpus.py` first "
            "(or `make gen-data`).")
    out: list[Document] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                out.append(Document(**json.loads(line)))
    return out


def load_questions(path: Path | None = None) -> list[Question]:
    path = path or settings.questions_path
    out: list[Question] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                out.append(Question(**json.loads(line)))
    return out
