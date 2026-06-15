"""Weight-space attestation — the deployable, exact form of Paramesphere `I(θ)` (S0).

This module fingerprints the *loaded* weights of the open-weight models this system runs
with full custody (the MiniLM embedder and the ms-marco cross-encoder reranker). It answers
one question the rest of the pipeline silently trusts: *are these the weights we think they
are, unchanged?* A tampered embedder corrupts every downstream retrieval without ever
raising an error, so this is a real attack surface, not ceremony.

What this is, precisely (truth-in-claims):

  * **Loaded-state fingerprint** — a deterministic, content-addressed digest computed over
    the model's in-memory tensor *values* (not its file bytes). It catches any change to
    the weights, including tampering applied *after* load that a file hash never sees.
  * **Re-serialization invariant** — re-saving / re-sharding identical weights changes the
    file bytes (so `sha256(file)` alarms) but not the values (so this fingerprint holds).
    That is the one case where a weight fingerprint genuinely beats a cheaper file hash.

What this is **not**: it is not the subspace `I(θ)` SVD of the Paramesphere research line,
and it is *not* robust to benign quantization — re-quantized weights have different values
and will trip it (a known false-positive mode, B2). For at-rest files, `artifact_sha256`
below is stronger and cheaper; the loaded-state fingerprint earns its keep only on the two
cases named above. Same-model tamper/swap on a fixed weight set — not cross-model identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "paramesphere-s0/1"


def _canonical_bytes(arr: np.ndarray) -> bytes:
    """Stable little-endian float32 byte view of a tensor's values.

    Canonicalising to `<f4` makes the digest invariant to storage details that do not
    change the values (C/F memory order, float64 inputs that equal their float32 form,
    non-contiguous views) while remaining sensitive to any change in the values themselves.
    """
    a = np.ascontiguousarray(np.asarray(arr), dtype="<f4")
    return a.tobytes()


def tensor_digest(name: str, arr: np.ndarray) -> str:
    """Content digest of one named tensor: its name, shape, and canonical values."""
    a = np.asarray(arr)
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(b"\x00")
    h.update(repr(tuple(int(d) for d in a.shape)).encode())
    h.update(b"\x00")
    h.update(_canonical_bytes(a))
    return h.hexdigest()


def loaded_state_fingerprint(named_tensors: dict[str, np.ndarray]) -> str:
    """A single `wfp:` digest over all tensors — order-independent, value-addressed."""
    h = hashlib.sha256()
    for name, dig in sorted(per_tensor_digests(named_tensors).items()):
        h.update(name.encode())
        h.update(b"\x00")
        h.update(dig.encode())
        h.update(b"\n")
    return "wfp:" + h.hexdigest()[:32]


def per_tensor_digests(named_tensors: dict[str, np.ndarray]) -> dict[str, str]:
    """Per-tensor digests — kept in the record so a tamper can be *localised* to a layer."""
    return {name: tensor_digest(name, arr) for name, arr in named_tensors.items()}


def param_count(named_tensors: dict[str, np.ndarray]) -> int:
    return int(sum(int(np.asarray(a).size) for a in named_tensors.values()))


def artifact_sha256(paths: list[Path]) -> str | None:
    """File-level hash of the at-rest weight artifacts (the cheaper, stronger at-rest pin).

    Returned alongside the loaded-state fingerprint so the contrast is explicit: this
    changes on any re-serialization; the fingerprint does not.
    """
    existing = sorted(p for p in paths if p.exists())
    if not existing:
        return None
    h = hashlib.sha256()
    for p in existing:
        h.update(p.name.encode())
        h.update(b"\x00")
        h.update(p.read_bytes())
        h.update(b"\n")
    return "sha256:" + h.hexdigest()[:32]


@dataclass(frozen=True)
class Attestation:
    """A signed-able record binding a model id to the fingerprint of its loaded weights."""

    model: str
    revision: str | None
    fingerprint: str          # wfp:... loaded-state, value-addressed
    artifact_sha256: str | None  # sha256:... at-rest file hash (re-serialization-sensitive)
    tensors: int
    params: int
    schema: str = SCHEMA_VERSION
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    per_tensor: dict[str, str] = field(default_factory=dict)

    def to_dict(self, *, include_per_tensor: bool = True) -> dict:
        d = asdict(self)
        if not include_per_tensor:
            d.pop("per_tensor", None)
        return d

    def to_json(self, *, include_per_tensor: bool = True) -> str:
        return json.dumps(self.to_dict(include_per_tensor=include_per_tensor),
                          indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> Attestation:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416 (explicit set)
        return cls(**{k: v for k, v in d.items() if k in known})


def attest(model: str, named_tensors: dict[str, np.ndarray], *,
           revision: str | None = None, artifact_paths: list[Path] | None = None) -> Attestation:
    """Build an :class:`Attestation` from a model id and its loaded named tensors."""
    return Attestation(
        model=model,
        revision=revision,
        fingerprint=loaded_state_fingerprint(named_tensors),
        artifact_sha256=artifact_sha256(artifact_paths) if artifact_paths else None,
        tensors=len(named_tensors),
        params=param_count(named_tensors),
        per_tensor=per_tensor_digests(named_tensors),
    )


@dataclass(frozen=True)
class VerifyResult:
    """The output of checking live weights against a committed baseline."""

    match: bool
    fingerprint_expected: str
    fingerprint_actual: str
    changed: list[str]   # tensor names whose digest diverged (localised tamper)
    added: list[str]
    removed: list[str]

    @property
    def ok(self) -> bool:
        return self.match


def verify(baseline: Attestation, named_tensors: dict[str, np.ndarray]) -> VerifyResult:
    """Compare live weights to a baseline; on mismatch, localise *which* tensors changed.

    Localisation needs the baseline's `per_tensor` digests; without them only the top-level
    fingerprint is compared (match/no-match with no per-layer detail).
    """
    actual_fp = loaded_state_fingerprint(named_tensors)
    actual = per_tensor_digests(named_tensors)
    base = baseline.per_tensor or {}
    changed = sorted(n for n in actual if n in base and actual[n] != base[n])
    added = sorted(n for n in actual if n not in base)
    removed = sorted(n for n in base if n not in actual)
    return VerifyResult(
        match=actual_fp == baseline.fingerprint,
        fingerprint_expected=baseline.fingerprint,
        fingerprint_actual=actual_fp,
        changed=changed,
        added=added if base else [],
        removed=removed if base else [],
    )


def named_tensors_from_state_dict(state_dict: dict) -> dict[str, np.ndarray]:
    """Adapter: a torch-style ``state_dict`` -> ``{name: float32 ndarray}``.

    Guarded so importing this module never pulls in torch; only called when a real model is
    handed in (the production path), never in the synthetic unit tests.
    """
    out: dict[str, np.ndarray] = {}
    for name, t in state_dict.items():
        if hasattr(t, "detach"):                       # torch.Tensor
            t = t.detach().to("cpu").float().numpy()
        out[name] = np.asarray(t, dtype=np.float32)
    return out


# Weight artifact filenames HF / safetensors / torch write, in order of preference.
_ARTIFACT_GLOBS = ("*.safetensors", "pytorch_model*.bin", "model*.bin")


def artifact_paths_from_dir(model_dir: str | Path | None) -> list[Path]:
    """The at-rest weight files under a resolved model directory (best-effort, graceful).

    Used at load to populate :func:`artifact_sha256` *when the model is present on disk* —
    the cheaper, stronger at-rest pin the spec asks for alongside the loaded-state
    fingerprint (S0, §10). Returns ``[]`` for a missing/unknown directory rather than
    raising, so a model loaded straight from the network (no local cache to hash) attests
    with ``artifact_sha256=null`` and the pipeline never crashes on a hashing failure.
    """
    if not model_dir:
        return []
    d = Path(model_dir)
    if not d.is_dir():
        return []
    for pattern in _ARTIFACT_GLOBS:
        hits = sorted(d.glob(pattern))
        if hits:
            return hits
    return []


def resolve_model_dir(model: object) -> Path | None:
    """Best-effort: the on-disk directory a loaded HF/sentence-transformers model came from.

    Probes the handful of attributes those libraries expose for the local path, and never
    raises — an unknown surface simply yields ``None`` (then ``artifact_sha256`` is ``null``).
    """
    for attr in ("model_card_data", "_model_config"):  # cheap no-ops if absent, kept explicit
        getattr(model, attr, None)
    candidates = [
        getattr(model, "_model_card_vars", {}).get("base_model") if isinstance(
            getattr(model, "_model_card_vars", None), dict) else None,
        getattr(model, "_model_dir", None),
        getattr(model, "model_path", None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
        getattr(getattr(model, "model", None), "name_or_path", None),
        getattr(model, "name_or_path", None),
    ]
    for c in candidates:
        try:
            if c and Path(str(c)).is_dir():
                return Path(str(c))
        except (TypeError, ValueError, OSError):  # pragma: no cover - defensive
            continue
    return None
