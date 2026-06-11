"""Attested run manifest — the binding claim, as executable assertions.

Synthetic and deterministic (no model download): a manifest binds weights → corpus → config
→ results into one content-addressed id, and that id moves iff one of those moves. The S0
attestations it composes are tested in test_attest.py; here we pin the composition.
"""
from __future__ import annotations

import numpy as np

from genealogy_rag.attest import attest
from genealogy_rag.manifest import (
    RunManifest,
    build_manifest,
    corpus_digest,
    model_ref,
    results_digest,
    verify_manifest,
)

CORPUS = [("d1", "Thomas Cason of Isle of Wight"), ("d2", "Ransom Cason of Marianna")]
RESULTS = {"hybrid": {"recall@5": 1.0}, "vector": {"recall@5": 0.4}}
CONFIG = {"ablations": ["vector", "hybrid"], "k_values": [1, 5]}


def _att(seed: int):
    rng = np.random.default_rng(seed)
    return attest(f"m{seed}", {"w": rng.standard_normal((8, 8)).astype(np.float32)})


def _manifest(*, corpus=CORPUS, results=RESULTS, config=CONFIG, seed=1):
    return build_manifest(
        [model_ref("embedder", _att(seed))],
        corpus=corpus_digest(corpus), corpus_docs=len(corpus),
        config=config, results=results_digest(results))


def test_digests_are_deterministic_and_prefixed():
    assert corpus_digest(CORPUS).startswith("cd:")
    assert results_digest(RESULTS).startswith("rd:")
    assert corpus_digest(CORPUS) == corpus_digest(list(reversed(CORPUS)))  # order-independent
    assert results_digest(RESULTS) == results_digest(dict(RESULTS))


def test_manifest_is_self_consistent_and_reproducible():
    m = _manifest()
    assert m.manifest_id.startswith("rm:") and verify_manifest(m)
    # the id excludes the timestamp, so two byte-identical runs share it
    assert _manifest().manifest_id == m.manifest_id


def test_a_swapped_weight_moves_the_id():
    base = _manifest(seed=1).manifest_id
    assert _manifest(seed=2).manifest_id != base  # different embedder fingerprint


def test_an_edited_corpus_moves_the_id():
    edited = [("d1", "Thomas Cason of Surrey"), ("d2", "Ransom Cason of Marianna")]
    assert _manifest(corpus=edited).manifest_id != _manifest().manifest_id


def test_a_changed_result_moves_the_id():
    assert _manifest(results={"hybrid": {"recall@5": 0.9}}).manifest_id != _manifest().manifest_id


def test_a_changed_config_moves_the_id():
    assert _manifest(config={"ablations": ["hybrid"]}).manifest_id != _manifest().manifest_id


def test_tampering_with_a_field_breaks_verification():
    m = _manifest()
    forged = RunManifest(
        manifest_id=m.manifest_id,                       # keep the old id
        models=m.models, corpus="cd:0000", corpus_docs=m.corpus_docs,  # but edit the corpus
        config=m.config, results=m.results)
    assert not verify_manifest(forged)


def test_unattested_model_carries_null_fingerprint_not_a_fake():
    m = build_manifest(
        [model_ref("reranker", None, model="cross-encoder/ms-marco")],
        corpus=corpus_digest(CORPUS), corpus_docs=2, config=CONFIG,
        results=results_digest(RESULTS))
    assert m.models[0].fingerprint is None and verify_manifest(m)


def test_json_roundtrip_is_stable():
    m = _manifest()
    assert '"manifest_id"' in m.to_json()
    assert m.to_dict()["corpus"] == m.corpus
