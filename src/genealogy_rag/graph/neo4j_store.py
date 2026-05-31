"""Neo4j graph backend -- the production path.

Implements the identical GraphStore interface as the in-memory store, but over a
real graph database using parameterized Cypher. Activated by setting NEO4J_URI
(see docker-compose.yml). Demonstrates: schema design (constraints + indexes),
variable-length traversal, and the query-tuning notes below.

  docker compose up -d neo4j
  NEO4J_URI=bolt://localhost:7687 python scripts/build_graph.py --load-neo4j
  NEO4J_URI=bolt://localhost:7687 make eval        # eval over the real DB

Query-tuning notes
------------------
* `:Person(id)` and `:Document(id)` carry UNIQUENESS constraints, which Neo4j
  backs with an index -- so MATCH-by-id is an O(1) seek, not a label scan.
* Neighbourhood expansion uses a bounded variable-length pattern
  `(seed)-[:CHILD_OF|PARENT_OF|SPOUSE_OF*..$hops]-(kin)`; bounding the upper hop
  count is what keeps the traversal from exploding on a dense graph.
* `MENTIONS` edges are pre-materialised (person<->document) so provenance lookups
  are single-hop expansions rather than full-text scans.
"""
from __future__ import annotations

from pathlib import Path

from ..config import settings
from .base import GraphStore, _raw_graph

SCHEMA = [
    "CREATE CONSTRAINT person_id IF NOT EXISTS "
    "FOR (p:Person) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
]


class Neo4jGraphStore(GraphStore):
    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None):
        from neo4j import GraphDatabase  # imported lazily; optional dependency
        self._driver = GraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    # ----- ingestion -------------------------------------------------------- #
    def ensure_schema(self) -> None:
        with self._driver.session() as s:
            for stmt in SCHEMA:
                s.run(stmt)

    def ingest(self, path: Path | None = None) -> None:
        """Idempotent bulk load of persons, kin edges, and MENTIONS edges."""
        raw = _raw_graph(path)
        self.ensure_schema()
        with self._driver.session() as s:
            s.run("UNWIND $rows AS r MERGE (p:Person {id: r.id}) "
                  "SET p.name = r.name, p.sex = r.sex",
                  rows=[{"id": p["id"], "name": p["name"], "sex": p.get("sex")}
                        for p in raw["persons"]])
            kin = [{"src": r["src"], "dst": r["dst"], "rel": r["rel"]}
                   for r in raw["relationships"]]
            # one MERGE per relationship type keeps the typed edges queryable
            for rel in ("CHILD_OF", "PARENT_OF", "SPOUSE_OF"):
                s.run(
                    f"UNWIND $rows AS r MATCH (a:Person {{id: r.src}}), "
                    f"(b:Person {{id: r.dst}}) MERGE (a)-[:{rel}]->(b)",
                    rows=[k for k in kin if k["rel"] == rel])
            edges = [{"person": p, "doc": d}
                     for p, ds in raw["person_documents"].items() for d in ds]
            s.run("UNWIND $rows AS r MERGE (d:Document {id: r.doc})", rows=edges)
            s.run("UNWIND $rows AS r MATCH (p:Person {id: r.person}), "
                  "(d:Document {id: r.doc}) MERGE (p)-[:MENTIONS]->(d)", rows=edges)

    def load(self, path: Path | None = None) -> Neo4jGraphStore:
        return self  # data is loaded once via ingest(); reads hit the live DB

    # ----- reads (GraphStore interface) ------------------------------------- #
    def persons_in_document(self, doc_id: str) -> list[str]:
        with self._driver.session() as s:
            rows = s.run("MATCH (p:Person)-[:MENTIONS]->(d:Document {id: $id}) "
                         "RETURN p.id AS id", id=doc_id)
            return [r["id"] for r in rows]

    def documents_for_person(self, person_id: str) -> list[str]:
        with self._driver.session() as s:
            rows = s.run("MATCH (p:Person {id: $id})-[:MENTIONS]->(d:Document) "
                         "RETURN d.id AS id", id=person_id)
            return [r["id"] for r in rows]

    def neighbourhood(self, person_ids: list[str], hops: int) -> dict[str, int]:
        with self._driver.session() as s:
            rows = s.run(
                "UNWIND $seeds AS sid "
                "MATCH (seed:Person {id: sid}) "
                "MATCH path = (seed)-[:CHILD_OF|PARENT_OF|SPOUSE_OF*0.." + str(int(hops))
                + "]-(kin:Person) "
                "RETURN kin.id AS id, min(length(path)) AS dist", seeds=person_ids)
            out: dict[str, int] = {}
            for r in rows:
                if r["id"] not in out or r["dist"] < out[r["id"]]:
                    out[r["id"]] = int(r["dist"])
            return out

    def person_name(self, person_id: str) -> str | None:
        with self._driver.session() as s:
            r = s.run("MATCH (p:Person {id: $id}) RETURN p.name AS n",
                      id=person_id).single()
            return r["n"] if r else None

    def all_person_names(self) -> dict[str, str]:
        with self._driver.session() as s:
            rows = s.run("MATCH (p:Person) RETURN p.id AS id, p.name AS n")
            return {r["n"].lower(): r["id"] for r in rows}

    def shortest_path(self, a: str, b: str) -> list[str]:
        with self._driver.session() as s:
            r = s.run(
                "MATCH (x:Person {id: $a}), (y:Person {id: $b}), "
                "p = shortestPath((x)-[:CHILD_OF|PARENT_OF|SPOUSE_OF*..12]-(y)) "
                "RETURN [n IN nodes(p) | n.id] AS ids", a=a, b=b).single()
            return r["ids"] if r else []

    def _neighbours_by_rel(self, person_id: str, rel: str) -> list[str]:
        with self._driver.session() as s:
            rows = s.run(
                f"MATCH (p:Person {{id: $id}})-[:{rel}]->(n:Person) RETURN n.id AS id",
                id=person_id)
            return [r["id"] for r in rows]

    def parents(self, person_id: str) -> list[str]:
        return self._neighbours_by_rel(person_id, "CHILD_OF")

    def children(self, person_id: str) -> list[str]:
        return self._neighbours_by_rel(person_id, "PARENT_OF")

    def spouses(self, person_id: str) -> list[str]:
        return self._neighbours_by_rel(person_id, "SPOUSE_OF")

    def sex(self, person_id: str) -> str | None:
        with self._driver.session() as s:
            r = s.run("MATCH (p:Person {id: $id}) RETURN p.sex AS sex",
                      id=person_id).single()
            return r["sex"] if r else None
