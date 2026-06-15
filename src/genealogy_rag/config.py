"""Central configuration. Everything tunable lives here or is overridable via env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # data
    documents_path: Path = ROOT / "data" / "genealogy" / "documents.jsonl"
    graph_path: Path = ROOT / "data" / "genealogy" / "graph.json"
    questions_path: Path = ROOT / "eval" / "questions.jsonl"
    cache_dir: Path = ROOT / ".cache"

    # models
    embed_model: str = field(default_factory=lambda: os.environ.get(
        "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    rerank_model: str = field(default_factory=lambda: os.environ.get(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    # Pinned HF revision (commit SHA) for each open-weight model (Paramesphere S0, §10).
    # Pin the exact commit at build via EMBED_REVISION / RERANK_REVISION so the loaded-state
    # fingerprint landed in attestation.json is bound to a known revision. Unset -> the
    # attestation records revision=null (the model resolves to whatever HF serves) and the
    # pipeline degrades gracefully — pinning is a tightening, never a load-time requirement.
    embed_revision: str | None = field(default_factory=lambda: os.environ.get("EMBED_REVISION"))
    rerank_revision: str | None = field(default_factory=lambda: os.environ.get("RERANK_REVISION"))
    embed_dim: int = 384

    # retrieval defaults
    rrf_k: int = 60          # reciprocal-rank-fusion constant
    graph_hops: int = 2      # neighbourhood radius for graph expansion
    graph_seed_docs: int = 5 # top fused docs whose entities seed the expansion
    rerank_candidates: int = 20

    # neo4j (only used when NEO4J_URI is set)
    neo4j_uri: str | None = field(default_factory=lambda: os.environ.get("NEO4J_URI"))
    neo4j_user: str = field(default_factory=lambda: os.environ.get("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.environ.get(
        "NEO4J_PASSWORD", "neo4jpassword"))


settings = Settings()
