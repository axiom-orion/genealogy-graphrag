#!/usr/bin/env python
"""Run the Scribe OCR eval and print the S2 production-gate verdict.

    python eval/run_scribe_eval.py                 # default: stub backends on the frozen corpus
    python eval/run_scribe_eval.py --backend noisy --error-rate 0.18
    python eval/run_scribe_eval.py --corpus data/scribe/corpus.jsonl --max-cer 0.08

The bundled corpus is **synthetic placeholder** text — these numbers measure the harness, not
a real model. Point `--corpus` at a frozen corpus of `/proof` artifacts with verified
transcriptions, run a real `OcrBackend`, and the same gate makes the self-hosting call.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genealogy_rag.scribe import (  # noqa: E402
    NoisyBackend,
    PerfectBackend,
    ScribeThresholds,
    evaluate,
    gates_production,
    load_corpus,
)

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "data" / "scribe" / "corpus.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--backend", choices=["perfect", "noisy", "both"], default="both")
    ap.add_argument("--error-rate", type=float, default=0.2)
    _d = ScribeThresholds()  # slots dataclass: read defaults off an instance, not the class
    ap.add_argument("--max-cer", type=float, default=_d.max_cer)
    ap.add_argument("--max-wer", type=float, default=_d.max_wer)
    ap.add_argument("--min-field-accuracy", type=float, default=_d.min_field_accuracy)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    thresholds = ScribeThresholds(max_cer=args.max_cer, max_wer=args.max_wer,
                                  min_field_accuracy=args.min_field_accuracy)
    backends = {"perfect": PerfectBackend(), "noisy": NoisyBackend(error_rate=args.error_rate)}
    chosen = backends.values() if args.backend == "both" else [backends[args.backend]]

    print(f"Scribe OCR eval — {len(corpus)} samples (synthetic placeholder corpus)")
    print(f"gate: CER ≤ {thresholds.max_cer:.2f} · WER ≤ {thresholds.max_wer:.2f} · "
          f"field ≥ {thresholds.min_field_accuracy:.2f}\n")
    for backend in chosen:
        report = evaluate(backend, corpus)
        gate = gates_production(report, thresholds)
        d = report.to_dict()
        print(f"  {report.backend:<14} CER {d['cer']:.3f} · WER {d['wer']:.3f} · "
              f"field {d['field_accuracy']:.3f} · per-hand {d['per_hand_cer']}")
        verdict = "PASSES the S2 gate" if gate.passed else "BLOCKED: " + "; ".join(gate.reasons)
        print(f"  {'':<14} → {verdict}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
