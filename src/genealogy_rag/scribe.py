"""Scribe — the OCR/extraction eval harness and the S2 production gate.

Scribe is the agent that turns a scanned record image into structured genealogical facts.
Self-hosting it (Vorion spec, S2) is justified only by **measured accuracy on a frozen eval
corpus**, never by enthusiasm — 17th-century secretary hand is brutal for every model, so the
corpus decides. This module is the decision machinery: it is model-agnostic, so a candidate
backend (TrOCR, a Qwen-VL-class document model, olmOCR, …) plugs in behind one protocol, and
the gate is the same regardless.

What is real and tested here:
  * **metrics** — character error rate (CER) and word error rate (WER) via Levenshtein, plus
    structured-field accuracy (did we recover the right name / date / place);
  * **the gate** — `gates_production(report, thresholds)`: a backend clears S2 only when CER
    and WER are at or below their caps AND field accuracy is at or above its floor;
  * **a frozen corpus format** + a deterministic synthetic corpus, so the harness runs in CI
    with no model and no download.

What is a documented seam, not a claim:
  * the real OCR backend (a self-hosted model) is an interface (`OcrBackend`) with a sketched
    adapter; running it is the human's infra step (Cloud Run GPU);
  * the **bundled corpus is synthetic placeholder** transcriptions — the real corpus is built
    from `/proof` artifacts with verified transcriptions, which only a human can assemble.
    The numbers this prints on synthetic data measure the *harness*, not a real model.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# --- the corpus ------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ScribeSample:
    """One frozen eval item: an image (by ref) and the verified ground truth it must yield."""

    id: str
    image_ref: str               # path/URI of the source artifact (not loaded by the harness)
    transcription: str           # the verified full-text transcription
    fields: Mapping[str, str] = field(default_factory=dict)  # name/date/place ground truth
    hand: str = "unknown"        # e.g. "print", "secretary", "italic" — difficulty stratum

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> ScribeSample:
        return cls(
            id=str(d["id"]),
            image_ref=str(d.get("image_ref", "")),
            transcription=str(d["transcription"]),
            fields={str(k): str(v) for k, v in dict(d.get("fields", {})).items()},
            hand=str(d.get("hand", "unknown")),
        )


def load_corpus(path: Path) -> list[ScribeSample]:
    """Load a frozen corpus (JSONL, one sample per line)."""
    out: list[ScribeSample] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(ScribeSample.from_dict(json.loads(line)))
    return out


# --- the backend seam ------------------------------------------------------- #


class OcrBackend:
    """A Scribe OCR backend: image bytes (or a ref) -> transcription, plus field extraction.

    Real backends (a self-hosted VLM) implement `transcribe`; the default `extract_fields`
    is a transparent regex pass over the transcription so a candidate is judged on its OCR,
    not on a bespoke parser. Override `extract_fields` to evaluate a model's own extraction.
    """

    name: str = "abstract"

    def transcribe(self, sample: ScribeSample) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def extract_fields(self, transcription: str) -> dict[str, str]:
        return default_field_extractor(transcription)


_DATE_RE = re.compile(
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b", re.I)
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_PLACE_RE = re.compile(r"\b(?:of|at|in)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})")
_NAME_RE = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b")


def default_field_extractor(text: str) -> dict[str, str]:
    """A deliberately transparent extractor: first name, date (or year), place mention."""
    out: dict[str, str] = {}
    if (m := _NAME_RE.search(text)):
        out["name"] = m.group(1)
    if (m := _DATE_RE.search(text)):
        out["date"] = m.group(1)
    elif (m := _YEAR_RE.search(text)):
        out["date"] = m.group(1)
    if (m := _PLACE_RE.search(text)):
        out["place"] = m.group(1)
    return out


# --- metrics ---------------------------------------------------------------- #


def levenshtein(a: Sequence[object], b: Sequence[object]) -> int:
    """Edit distance over any two sequences (characters for CER, tokens for WER)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(prediction: str, reference: str) -> float:
    """Character error rate = edit distance / |reference| (0.0 = perfect, 1.0 if empty ref)."""
    if not reference:
        return 0.0 if not prediction else 1.0
    return levenshtein(prediction, reference) / len(reference)


def wer(prediction: str, reference: str) -> float:
    """Word error rate = token edit distance / #reference tokens."""
    ref_tokens = reference.split()
    if not ref_tokens:
        return 0.0 if not prediction.split() else 1.0
    return levenshtein(prediction.split(), ref_tokens) / len(ref_tokens)


def field_accuracy(predicted: Mapping[str, str], truth: Mapping[str, str]) -> float:
    """Share of ground-truth fields recovered exactly (case-insensitive, trimmed)."""
    if not truth:
        return 1.0
    hits = sum(1 for k, v in truth.items()
               if k in predicted and predicted[k].strip().casefold() == v.strip().casefold())
    return hits / len(truth)


# --- the report + the gate -------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ScribeThresholds:
    """The S2 bar. Defaults are a starting policy, not a measurement — set them from the
    verified corpus once a real backend runs against it."""

    max_cer: float = 0.10
    max_wer: float = 0.15
    min_field_accuracy: float = 0.90


@dataclass(frozen=True, slots=True)
class ScribeReport:
    backend: str
    samples: int
    cer: float
    wer: float
    field_accuracy: float
    per_hand: Mapping[str, float]   # mean CER by hand stratum (where the difficulty hides)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "samples": self.samples, "cer": round(self.cer, 4),
                "wer": round(self.wer, 4), "field_accuracy": round(self.field_accuracy, 4),
                "per_hand_cer": {k: round(v, 4) for k, v in self.per_hand.items()}}


def evaluate(backend: OcrBackend, corpus: Iterable[ScribeSample]) -> ScribeReport:
    """Run a backend over the corpus and aggregate CER / WER / field accuracy."""
    samples = list(corpus)
    if not samples:
        return ScribeReport(backend.name, 0, 0.0, 0.0, 0.0, {})
    cers, wers, faccs = [], [], []
    by_hand: dict[str, list[float]] = {}
    for s in samples:
        pred = backend.transcribe(s)
        c = cer(pred, s.transcription)
        cers.append(c)
        wers.append(wer(pred, s.transcription))
        faccs.append(field_accuracy(backend.extract_fields(pred), s.fields))
        by_hand.setdefault(s.hand, []).append(c)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731 - terse, local
    return ScribeReport(
        backend=backend.name, samples=len(samples),
        cer=mean(cers), wer=mean(wers), field_accuracy=mean(faccs),
        per_hand={h: mean(v) for h, v in sorted(by_hand.items())})


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reasons: list[str]   # the failing clauses; empty when passed


def gates_production(report: ScribeReport,
                     thresholds: ScribeThresholds | None = None) -> GateResult:
    """The S2 decision: does measured accuracy justify self-hosting this backend?"""
    t = thresholds or ScribeThresholds()
    reasons: list[str] = []
    if report.samples == 0:
        reasons.append("empty corpus — nothing measured")
    if report.cer > t.max_cer:
        reasons.append(f"CER {report.cer:.3f} > {t.max_cer:.3f}")
    if report.wer > t.max_wer:
        reasons.append(f"WER {report.wer:.3f} > {t.max_wer:.3f}")
    if report.field_accuracy < t.min_field_accuracy:
        reasons.append(f"field accuracy {report.field_accuracy:.3f} < {t.min_field_accuracy:.3f}")
    return GateResult(passed=not reasons, reasons=reasons)


# --- reference backends ----------------------------------------------------- #


class PerfectBackend(OcrBackend):
    """Returns the ground truth — CER/WER 0, gate passes. Proves the harness's happy path."""

    name = "perfect-stub"

    def transcribe(self, sample: ScribeSample) -> str:
        return sample.transcription


class NoisyBackend(OcrBackend):
    """Deterministically corrupts the reference to a target character error rate, so the GATE
    can be exercised both ways without a real model. Seeded — identical runs, identical CER."""

    name = "noisy-stub"

    def __init__(self, error_rate: float = 0.2, seed: int = 7):
        self.error_rate = error_rate
        self._seed = seed

    def transcribe(self, sample: ScribeSample) -> str:
        import random
        rng = random.Random(f"{self._seed}:{sample.id}")
        chars = list(sample.transcription)
        for i in range(len(chars)):
            if rng.random() < self.error_rate:
                chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz ")
        return "".join(chars)


# A self-hosted VLM adapter would live here, behind a guarded import, e.g.:
#
#   class TrOcrBackend(OcrBackend):                       # pragma: no cover - infra path
#       name = "trocr-handwritten"
#       def __init__(self, model_id: str): ...            # loads weights -> attest via S0
#       def transcribe(self, sample): return self._infer(load_image(sample.image_ref))
#
# Its weights are fingerprinted by genealogy_rag.attest (S0) before it is trusted; running
# it is the Cloud Run GPU step, gated on this harness clearing ScribeThresholds.
