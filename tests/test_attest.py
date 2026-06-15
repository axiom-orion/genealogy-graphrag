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
    artifact_paths_from_dir,
    artifact_sha256,
    attest,
    loaded_state_fingerprint,
    named_tensors_from_state_dict,
    param_count,
    resolve_model_dir,
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


# --- S0 §10 record: pinned HF revision + at-rest artifact SHA-256, populated at load -------

def _write_weight_dir(tmp_dir, m, *, fmt="bin", extra=False):
    """Write a synthetic model dir with HF-shaped weight files (no torch/safetensors needed).

    The filenames are what attest's artifact-glob looks for; the contents are arbitrary bytes
    standing in for the at-rest artifact, so `artifact_sha256` has something real to hash.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    name = "pytorch_model.bin" if fmt == "bin" else "model.safetensors"
    payload = b"".join(np.ascontiguousarray(v, dtype="<f4").tobytes() for v in m.values())
    if extra:
        payload += b"\x00re-serialized-padding"   # same weights, different at-rest bytes
    (tmp_dir / name).write_bytes(payload)
    return tmp_dir / name


def test_artifact_paths_resolution_degrades_gracefully(tmp_path):
    # missing / None / non-dir -> [] (never raises): a network-served model with no local
    # cache attests with artifact_sha256=null, and the pipeline keeps running.
    assert artifact_paths_from_dir(None) == []
    assert artifact_paths_from_dir(tmp_path / "does-not-exist") == []
    f = tmp_path / "afile"
    f.write_text("x")
    assert artifact_paths_from_dir(f) == []                      # a file, not a dir
    assert artifact_paths_from_dir(tmp_path) == []               # empty dir, no weight files


def test_artifact_paths_finds_weight_files_when_present(tmp_path):
    m = _model()
    _write_weight_dir(tmp_path / "mdl", m, fmt="safetensors")
    found = artifact_paths_from_dir(tmp_path / "mdl")
    assert [p.name for p in found] == ["model.safetensors"]
    assert artifact_sha256(found) is not None                    # hashable at-rest artifact


def test_resolve_model_dir_finds_dir_via_config_and_is_none_otherwise(tmp_path):
    class _Cfg:
        _name_or_path = str(tmp_path)
    class _ModelWithConfig:
        config = _Cfg()
    assert resolve_model_dir(_ModelWithConfig()) == tmp_path     # found via config._name_or_path
    assert resolve_model_dir(object()) is None                   # unknown surface -> None, no raise


def test_attest_lands_revision_and_artifact_sha_at_load(tmp_path):
    """The §10 S0 record: one Attestation carrying revision + artifact SHA-256 + fingerprint."""
    m = _model()
    _write_weight_dir(tmp_path / "mdl", m)
    rec = attest("embedder", m, revision="a1b2c3d",
                 artifact_paths=artifact_paths_from_dir(tmp_path / "mdl"))
    assert rec.revision == "a1b2c3d"                             # pinned commit recorded
    assert rec.fingerprint.startswith("wfp:")                    # loaded-state fingerprint
    assert rec.artifact_sha256 and rec.artifact_sha256.startswith("sha256:")  # at-rest pin
    # all three survive the JSON round-trip that lands them in attestation.json
    back = Attestation.from_dict(rec.to_dict())
    assert (back.revision, back.fingerprint, back.artifact_sha256) == (
        rec.revision, rec.fingerprint, rec.artifact_sha256)


def test_reserialization_holds_fingerprint_while_artifact_sha_moves(tmp_path):
    """The headline S0 case, end-to-end through the load-time helpers (§4-4 (a)+(b)).

    Re-serialize the *same weights* to a fresh artifact (different bytes). The at-rest
    artifact_sha256 — a plain file hash — alarms; the loaded-state fingerprint correctly says
    'same model'. Then a single-value tamper trips the fingerprint, localised to one tensor.
    """
    m = _model()
    d1, d2 = tmp_path / "v1", tmp_path / "v2"
    _write_weight_dir(d1, m)
    _write_weight_dir(d2, m, extra=True)                         # identical weights, new bytes

    base = attest("embedder", m, revision="r1",
                  artifact_paths=artifact_paths_from_dir(d1))
    reser = attest("embedder", m, revision="r1",
                   artifact_paths=artifact_paths_from_dir(d2))

    # (a) survives benign re-serialization: fingerprint identical, file hash differs
    assert reser.fingerprint == base.fingerprint                 # 'same model' — correct
    assert reser.artifact_sha256 != base.artifact_sha256         # file hash false-alarms

    # (b) trips on tamper, localised to the changed layer (what a file hash cannot localise)
    tampered = {k: v.copy() for k, v in m.items()}
    tampered["embeddings.weight"][1, 1] += np.float32(1e-3)
    res = verify(base, tampered)
    assert not res.ok and res.changed == ["embeddings.weight"]
