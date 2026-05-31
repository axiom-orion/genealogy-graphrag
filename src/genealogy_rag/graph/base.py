"""Backend-agnostic kinship-graph interface used by the retriever."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import settings

KIN_RELS = ("CHILD_OF", "PARENT_OF", "SPOUSE_OF")


def _raw_graph(path: Path | None = None) -> dict:
    return json.loads((path or settings.graph_path).read_text())


class GraphStore(ABC):
    """Minimal surface the hybrid retriever needs from the graph layer."""

    @abstractmethod
    def load(self, path: Path | None = None) -> GraphStore: ...

    @abstractmethod
    def persons_in_document(self, doc_id: str) -> list[str]: ...

    @abstractmethod
    def documents_for_person(self, person_id: str) -> list[str]: ...

    @abstractmethod
    def neighbourhood(self, person_ids: list[str], hops: int) -> dict[str, int]:
        """person_id -> shortest hop distance, for all kin within `hops` of any seed."""

    @abstractmethod
    def person_name(self, person_id: str) -> str | None: ...

    @abstractmethod
    def all_person_names(self) -> dict[str, str]:
        """name (lowercased) -> person_id, for query-side entity linking."""

    @abstractmethod
    def shortest_path(self, a: str, b: str) -> list[str]: ...

    # --- typed kinship navigation (used by the relation resolver) ---------- #
    @abstractmethod
    def parents(self, person_id: str) -> list[str]: ...

    @abstractmethod
    def children(self, person_id: str) -> list[str]: ...

    @abstractmethod
    def spouses(self, person_id: str) -> list[str]: ...

    @abstractmethod
    def sex(self, person_id: str) -> str | None: ...
