"""Attested run manifest — bind the weights to the corpus to the result.

A retrieval number is only as trustworthy as the chain behind it. This module produces a
single content-addressed record that answers, for a published eval result, *which weights*
(by their S0 loaded-state fingerprint) ran over *which corpus* (by content hash) under
*which configuration* to produce *which results* (by digest). Change any one — a swapped
embedder, an edited document, a tweaked ablation, a different score — and the `manifest_id`
moves, so a result can be re-derived and checked rather than taken on faith.

It composes the S0 weight attestation (`attest.py`) with corpus + config + results digests;
it is pure and deterministic, and claims nothing the inputs don't support — a manifest over
unfingerprinted models simply carries `null` fingerprints and says so.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .attest import Attestation

MANIFEST_SCHEMA = "attested-run/1"


def _sha(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def corpus_digest(documents: Iterable[tuple[str, str]]) -> str:
    """Content hash of a corpus: sorted (id, searchable-text) pairs — order-independent,
    sensitive to any edit of any document."""
    pairs = sorted((str(i), str(t)) for i, t in documents)
    return "cd:" + _sha(pairs)[:32]


def results_digest(results: object) -> str:
    """Content hash of an eval result object (the metrics, however nested)."""
    return "rd:" + _sha(results)[:32]


@dataclass(frozen=True)
class ModelRef:
    """One model in the run, by role, at the attestation grade actually obtained."""

    role: str                    # e.g. "embedder" | "reranker"
    model: str
    fingerprint: str | None      # wfp:... loaded-state, or null when unattested
    params: int | None = None


def model_ref(role: str, attestation: Attestation | None, *, model: str | None = None) -> ModelRef:
    """A ModelRef from an S0 Attestation (or a bare declared model when none is available)."""
    if attestation is None:
        return ModelRef(role=role, model=model or "(unattested)", fingerprint=None)
    return ModelRef(role=role, model=attestation.model,
                    fingerprint=attestation.fingerprint, params=attestation.params)


@dataclass(frozen=True)
class RunManifest:
    manifest_id: str             # rm:... hash over models + corpus + config + results
    models: list[ModelRef]
    corpus: str                  # cd:...
    corpus_docs: int
    config: Mapping[str, object]
    results: str                 # rd:...
    schema: str = MANIFEST_SCHEMA
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def build_manifest(models: Iterable[ModelRef], *, corpus: str, corpus_docs: int,
                   config: Mapping[str, object], results: str) -> RunManifest:
    """Compose the content-addressed manifest. `manifest_id` excludes the timestamp, so
    two byte-identical runs share an id (reproducibility), while any substantive change to
    weights / corpus / config / results moves it."""
    model_list = sorted(models, key=lambda m: (m.role, m.model))
    body = {
        "models": [{"role": m.role, "model": m.model, "fingerprint": m.fingerprint}
                   for m in model_list],
        "corpus": corpus,
        "corpus_docs": corpus_docs,
        "config": config,
        "results": results,
        "schema": MANIFEST_SCHEMA,
    }
    return RunManifest(
        manifest_id="rm:" + _sha(body)[:32],
        models=model_list, corpus=corpus, corpus_docs=corpus_docs,
        config=dict(config), results=results)


def verify_manifest(manifest: RunManifest) -> bool:
    """Recompute the id from the manifest's own fields — True iff it is internally consistent
    (i.e. nobody edited a field without re-deriving the id)."""
    body = {
        "models": [{"role": m.role, "model": m.model, "fingerprint": m.fingerprint}
                   for m in sorted(manifest.models, key=lambda m: (m.role, m.model))],
        "corpus": manifest.corpus,
        "corpus_docs": manifest.corpus_docs,
        "config": dict(manifest.config),
        "results": manifest.results,
        "schema": manifest.schema,
    }
    return manifest.manifest_id == "rm:" + _sha(body)[:32]
