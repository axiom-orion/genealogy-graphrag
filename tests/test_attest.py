"""Weight-space attestation (Paramesphere S0) — the claims, as executable assertions.

These run on synthetic numpy tensors: no model download, deterministic, offline — the same
discipline as the rest of the suite. They pin the two properties that justify a weight
fingerprint over a file hash, and the tamper-localisation the demo and drift-audit rely on.
"""
from __future__ import annotations

import numpy as np
import pytest

from genealogy_rag.attest import (
    Attestation,
    artifact_sha256,
    attest,
    loaded_state_fingerprint,
    named_tensors_from_state_dict,
    param_count,
    tensor_digest,
    verify,
)


def _model(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "layer.0.weight": rng.standard_normal((8, 8)).astype(np.float32),
        "layer.0.bias": rng.standard_normal(8).astype(np.float32),
        "embeddings.weight": rng.standard_normal((16, 8)).astype(np.float32),
    }


def test_fingerprint_is_deterministic_and_prefixed():
    m = _model()
    fp = loaded_state_fingerprint(m)
    assert fp.startswith("wfp:")
    assert fp == loaded_state_fingerprint(m)               # stable across calls


def test_fingerprint_is_order_independent():
    m = _model()
    shuffled = {k: m[k] for k in reversed(list(m))}
    assert loaded_state_fingerprint(shuffled) == loaded_state_fingerprint(m)


def test_fingerprint_is_invariant_to_storage_not_values():
    m = _model()
    fp = loaded_state_fingerprint(m)
    # float64 inputs equal to their float32 form, and non-contiguous views, must not move it
    same_values = {
        k: np.asarray(v, dtype=np.float64) for k, v in m.items()  # wider dtype, same values
    }
    same_values["layer.0.weight"] = np.ascontiguousarray(same_values["layer.0.weight"].T).T
    assert loaded_state_fingerprint(same_values) == fp
    # but a real change to a single value must move it
    changed = {k: v.copy() for k, v in m.items()}
    changed["layer.0.bias"][0] += np.float32(1e-4)
    assert loaded_state_fingerprint(changed) != fp


def test_tamper_is_caught_and_localised():
    m = _model()
    base = attest("m", m, revision="r1")
    tampered = {k: v.copy() for k, v in m.items()}
    tampered["embeddings.weight"][3, 2] += np.float32(0.01)
    res = verify(base, tampered)
    assert not res.match and not res.ok
    assert res.changed == ["embeddings.weight"]            # localised to the one layer
    assert res.added == [] and res.removed == []


def test_added_and_removed_tensors_are_reported():
    m = _model()
    base = attest("m", m)
    grown = {**{k: v.copy() for k, v in m.items()}, "layer.1.weight": np.zeros((4, 4), np.float32)}
    del grown["layer.0.bias"]
    res = verify(base, grown)
    assert res.added == ["layer.1.weight"]
    assert res.removed == ["layer.0.bias"]


def test_clean_reload_verifies():
    m = _model()
    base = attest("m", m)
    identical = {k: v.copy() for k, v in m.items()}
    assert verify(base, identical).match


def test_reserialization_survives_fingerprint_but_not_file_hash(tmp_path):
    """The one case a weight fingerprint beats sha256: same values, different file bytes."""
    m = _model()
    fp = loaded_state_fingerprint(m)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, **m)
    np.savez(b, _meta=np.array([1, 2, 3]), **m)            # same weights, extra array -> new bytes
    reloaded = {k: np.load(a)[k] for k in m}
    assert loaded_state_fingerprint(reloaded) == fp        # values preserved -> fingerprint holds
    assert artifact_sha256([a]) != artifact_sha256([b])    # file bytes changed -> file hash moves


def test_counts():
    m = _model()
    a = attest("m", m)
    assert a.tensors == 3
    assert a.params == param_count(m) == 8 * 8 + 8 + 16 * 8


def test_record_json_roundtrip():
    a = attest("m", _model(), revision="rev-abc")
    restored = Attestation.from_dict(a.to_dict())
    assert restored.fingerprint == a.fingerprint
    assert restored.model == "m" and restored.revision == "rev-abc"
    assert "fingerprint" in a.to_json()
    # per_tensor can be dropped for a compact on-disk record
    assert "per_tensor" not in a.to_dict(include_per_tensor=False)


def test_tensor_digest_changes_with_name_and_shape():
    arr = np.ones((4, 4), np.float32)
    assert tensor_digest("a", arr) != tensor_digest("b", arr)         # name matters
    assert tensor_digest("a", arr) != tensor_digest("a", arr.reshape(2, 8))  # shape matters


def test_state_dict_adapter_handles_torch_like_and_numpy():
    class FakeTensor:
        """Mimics the torch.Tensor surface the adapter touches."""
        def __init__(self, a): self._a = a
        def detach(self): return self
        def to(self, _device): return self
        def float(self): return self
        def numpy(self): return self._a

    sd = {"w": FakeTensor(np.ones((2, 2), np.float32)), "b": np.zeros(2, np.float32)}
    nt = named_tensors_from_state_dict(sd)
    assert set(nt) == {"w", "b"}
    assert all(v.dtype == np.float32 for v in nt.values())
    assert loaded_state_fingerprint(nt)            # usable downstream


def test_artifact_sha256_none_when_absent(tmp_path):
    assert artifact_sha256([tmp_path / "nope.bin"]) is None


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_distinct_models_distinct_fingerprints(seed):
    assert loaded_state_fingerprint(_model(seed)) != loaded_state_fingerprint(_model(seed + 100))
