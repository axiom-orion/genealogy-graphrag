#!/usr/bin/env python
"""Emit a weight-space attestation for the open-weight models this system runs (S0).

    python scripts/attest_weights.py            # attest the real embedder + reranker
    python scripts/attest_weights.py --out attestation/weights.json
    python scripts/attest_weights.py --demo     # synthetic, no model download (CI-safe)

`--demo` builds two deterministic synthetic "models", attests them, and runs the two
self-checks that justify a weight fingerprint at all (exiting non-zero if either fails):

  1. tamper is caught and localised — perturb one weight and `verify` names the layer;
  2. re-serialization is survived — the same values written to different bytes keep the
     loaded-state fingerprint while the at-rest file hash changes.

The real mode loads the configured models (a download on first run) and writes their
attestation; the existing weekly drift-audit can then regression-check the fingerprints.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genealogy_rag.attest import (  # noqa: E402
    Attestation,
    artifact_sha256,
    attest,
    loaded_state_fingerprint,
    verify,
)


def _synthetic_model(seed: int, layers: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    nt: dict[str, np.ndarray] = {}
    for i in range(layers):
        nt[f"encoder.layer.{i}.weight"] = rng.standard_normal((64, 64)).astype(np.float32)
        nt[f"encoder.layer.{i}.bias"] = rng.standard_normal(64).astype(np.float32)
    nt["embeddings.word_embeddings.weight"] = rng.standard_normal((512, 64)).astype(np.float32)
    return nt


def _run_demo(out: Path | None) -> int:
    embedder = _synthetic_model(seed=1)
    reranker = _synthetic_model(seed=2, layers=4)
    att_e = attest("synthetic/embedder", embedder, revision="demo")
    att_r = attest("synthetic/reranker", reranker, revision="demo")
    print(f"embedder  {att_e.fingerprint}  ({att_e.tensors} tensors, {att_e.params:,} params)")
    print(f"reranker  {att_r.fingerprint}  ({att_r.tensors} tensors, {att_r.params:,} params)")

    failures = 0

    # check 1 — tamper is caught and localised
    tampered = {k: v.copy() for k, v in embedder.items()}
    tampered["encoder.layer.3.weight"][0, 0] += np.float32(1e-3)
    res = verify(att_e, tampered)
    if res.match or res.changed != ["encoder.layer.3.weight"]:
        print(f"  FAIL: tamper not localised (match={res.match}, changed={res.changed})")
        failures += 1
    else:
        print(f"  ok: tamper localised to {res.changed[0]}")

    # check 2 — re-serialization survives the fingerprint but not the file hash
    fp_before = loaded_state_fingerprint(embedder)
    a, b = Path("/tmp/attest_a.npz"), Path("/tmp/attest_b.npz")
    np.savez(a, **embedder)
    np.savez(b, _meta=np.array([1, 2, 3]), **embedder)   # same values, different bytes
    reloaded = {k: np.load(a)[k] for k in embedder}
    fp_after = loaded_state_fingerprint(reloaded)
    file_changed = artifact_sha256([a]) != artifact_sha256([b])
    a.unlink(missing_ok=True)
    b.unlink(missing_ok=True)
    if fp_before != fp_after or not file_changed:
        print(f"  FAIL: re-serialization invariance (fp_stable={fp_before == fp_after}, "
              f"file_changed={file_changed})")
        failures += 1
    else:
        print("  ok: fingerprint stable across re-serialization; file hash changed")

    if out:
        _write(out, [att_e, att_r])
    return 1 if failures else 0


def _run_real(out: Path) -> int:
    from genealogy_rag.embeddings import Embedder
    from genealogy_rag.rerank import CrossEncoderReranker
    records = [Embedder().attest()]
    r = CrossEncoderReranker().attest()
    if r is not None:
        records.append(r)
    for rec in records:
        print(f"{rec.model}  {rec.fingerprint}  ({rec.tensors} tensors, {rec.params:,} params)")
    _write(out, records)
    return 0


def _write(out: Path, records: list[Attestation]) -> None:
    import json
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": records[0].schema if records else "paramesphere-s0/1",
               "models": [r.to_dict(include_per_tensor=False) for r in records]}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="synthetic, no model download")
    ap.add_argument("--out", type=Path, default=None, help="write attestation JSON here")
    args = ap.parse_args()
    if args.demo:
        return _run_demo(args.out)
    default_out = Path(__file__).resolve().parents[1] / "attestation" / "weights.json"
    return _run_real(args.out or default_out)


if __name__ == "__main__":
    raise SystemExit(main())
