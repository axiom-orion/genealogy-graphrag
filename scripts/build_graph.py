"""Build/inspect the kinship graph; optionally ingest it into Neo4j.

  python scripts/build_graph.py                 # stats (in-memory NetworkX)
  NEO4J_URI=bolt://localhost:7687 \
      python scripts/build_graph.py --load-neo4j  # ingest into a live Neo4j
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genealogy_rag.graph import NetworkXGraphStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-neo4j", action="store_true",
                    help="ingest the graph into the Neo4j at $NEO4J_URI")
    args = ap.parse_args()

    if args.load_neo4j:
        from genealogy_rag.graph.neo4j_store import Neo4jGraphStore
        store = Neo4jGraphStore()
        store.ingest()
        print("ingested graph into Neo4j; sample maternal-grandfather query:")
        # P21 = Nancy Ainsworth -> mother -> father
        path = store.shortest_path("P21", "P05")
        print("  shortestPath(P21, P05) =", path)
        store.close()
        return

    g = NetworkXGraphStore().load()
    n_persons = g.kin.number_of_nodes()
    n_edges = g.directed.number_of_edges()
    print(f"persons: {n_persons} | directed kin edges: {n_edges}")
    # demonstrate typed navigation
    nancy = "P21"
    print(f"Nancy Ainsworth parents: {[g.person_name(p) for p in g.parents(nancy)]}")
    print(f"shortest path Nancy -> Thomas Calloway: "
          f"{[g.person_name(p) for p in g.shortest_path('P21', 'P05')]}")


if __name__ == "__main__":
    main()
