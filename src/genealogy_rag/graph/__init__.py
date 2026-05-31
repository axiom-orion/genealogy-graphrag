from .base import GraphStore
from .networkx_store import NetworkXGraphStore

__all__ = ["GraphStore", "NetworkXGraphStore", "load_graph_store"]


def load_graph_store():
    """Return a Neo4j-backed store if NEO4J_URI is set, else the in-memory store.

    Both implement the identical GraphStore interface, so retrieval code is
    backend-agnostic -- the eval runs on NetworkX in CI; production points at Neo4j."""
    from ..config import settings
    if settings.neo4j_uri:
        from .neo4j_store import Neo4jGraphStore
        return Neo4jGraphStore()
    return NetworkXGraphStore()
