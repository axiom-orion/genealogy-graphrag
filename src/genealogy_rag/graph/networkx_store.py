"""In-memory graph backend (NetworkX). Zero external dependencies at runtime, so a
reviewer can clone and `make eval` with no database. Same interface as Neo4j store."""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from .base import GraphStore, _raw_graph


class NetworkXGraphStore(GraphStore):
    def __init__(self):
        self.kin = nx.Graph()           # undirected kin graph for traversal
        self.directed = nx.DiGraph()    # keeps CHILD_OF / PARENT_OF direction
        self._names: dict[str, str] = {}
        self._attrs: dict[str, dict] = {}
        self._person_docs: dict[str, list[str]] = {}
        self._doc_persons: dict[str, list[str]] = {}

    def load(self, path: Path | None = None) -> NetworkXGraphStore:
        raw = _raw_graph(path)
        for p in raw["persons"]:
            self._names[p["id"]] = p["name"]
            self._attrs[p["id"]] = p
            self.kin.add_node(p["id"])
            self.directed.add_node(p["id"])
        for r in raw["relationships"]:
            self.kin.add_edge(r["src"], r["dst"], rel=r["rel"])
            self.directed.add_edge(r["src"], r["dst"], rel=r["rel"])
        self._person_docs = raw["person_documents"]
        for person, ds in self._person_docs.items():
            for d in ds:
                self._doc_persons.setdefault(d, []).append(person)
        return self

    def persons_in_document(self, doc_id: str) -> list[str]:
        return list(self._doc_persons.get(doc_id, []))

    def documents_for_person(self, person_id: str) -> list[str]:
        return list(self._person_docs.get(person_id, []))

    def neighbourhood(self, person_ids: list[str], hops: int) -> dict[str, int]:
        out: dict[str, int] = {}
        for seed in person_ids:
            if seed not in self.kin:
                continue
            for node, dist in nx.single_source_shortest_path_length(
                    self.kin, seed, cutoff=hops).items():
                if node not in out or dist < out[node]:
                    out[node] = dist
        return out

    def person_name(self, person_id: str) -> str | None:
        return self._names.get(person_id)

    def all_person_names(self) -> dict[str, str]:
        return {n.lower(): pid for pid, n in self._names.items()}

    def shortest_path(self, a: str, b: str) -> list[str]:
        try:
            return nx.shortest_path(self.kin, a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _out(self, pid: str, rel: str) -> list[str]:
        if pid not in self.directed:
            return []
        return [v for _, v, d in self.directed.out_edges(pid, data=True)
                if d.get("rel") == rel]

    def parents(self, person_id: str) -> list[str]:
        return self._out(person_id, "CHILD_OF")

    def children(self, person_id: str) -> list[str]:
        return self._out(person_id, "PARENT_OF")

    def spouses(self, person_id: str) -> list[str]:
        return self._out(person_id, "SPOUSE_OF")

    def sex(self, person_id: str) -> str | None:
        return self._attrs.get(person_id, {}).get("sex")
