"""Scribe OCR eval harness + S2 gate — the decision machinery, as executable assertions.

Synthetic and deterministic (no model, no download), the same discipline as the rest of the
suite. These pin the metrics and the gate; the real OCR backend and a verified `/proof`
corpus are the human's infra step.
"""
from __future__ import annotations

from pathlib import Path

from genealogy_rag.scribe import (
    NoisyBackend,
    PerfectBackend,
    ScribeSample,
    ScribeThresholds,
    cer,
    default_field_extractor,
    evaluate,
    field_accuracy,
    gates_production,
    levenshtein,
    load_corpus,
    wer,
)

CORPUS = Path(__file__).resolve().parents[1] / "data" / "scribe" / "corpus.jsonl"


# --- metrics ---------------------------------------------------------------- #

def test_levenshtein_basics():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("", "abc") == 3
    assert levenshtein(["a", "b"], ["a", "b", "c"]) == 1


def test_cer_and_wer_perfect_and_empty():
    assert cer("abc", "abc") == 0.0
    assert wer("a b c", "a b c") == 0.0
    assert cer("", "abc") == 1.0
    assert cer("", "") == 0.0
    # one char wrong out of four -> CER 0.25
    assert abs(cer("abXd", "abcd") - 0.25) < 1e-9
    # one word wrong out of three -> WER ~0.333
    assert abs(wer("the quick dog", "the quick fox") - 1 / 3) < 1e-9


def test_field_accuracy_is_case_insensitive_and_trimmed():
    truth = {"name": "Thomas Cason", "date": "1709"}
    assert field_accuracy({"name": "  thomas cason ", "date": "1709"}, truth) == 1.0
    assert field_accuracy({"name": "Thomas Cason"}, truth) == 0.5
    assert field_accuracy({}, {}) == 1.0  # nothing to recover


def test_default_extractor_pulls_name_date_place():
    f = default_field_extractor("Ransom Cason departed on 12 Nov 1853 at Marianna")
    assert f["name"] == "Ransom Cason"
    assert f["date"] == "12 Nov 1853"
    assert f["place"] == "Marianna"
    # falls back to a bare year when no full date is present
    assert default_field_extractor("born in 1822")["date"] == "1822"


# --- the corpus ------------------------------------------------------------- #

def test_frozen_corpus_loads_and_is_well_formed():
    corpus = load_corpus(CORPUS)
    assert len(corpus) >= 5
    assert all(isinstance(s, ScribeSample) and s.transcription for s in corpus)
    assert {s.hand for s in corpus} & {"secretary", "print"}  # difficulty strata present


# --- the gate, both ways ---------------------------------------------------- #

def test_perfect_backend_passes_the_gate():
    report = evaluate(PerfectBackend(), load_corpus(CORPUS))
    assert report.cer == 0.0 and report.wer == 0.0 and report.field_accuracy == 1.0
    assert gates_production(report).passed


def test_noisy_backend_is_blocked_with_reasons():
    report = evaluate(NoisyBackend(error_rate=0.3, seed=1), load_corpus(CORPUS))
    assert report.cer > 0.0
    gate = gates_production(report)
    assert not gate.passed and gate.reasons  # named failing clauses, not a bare False


def test_noisy_backend_is_deterministic():
    corpus = load_corpus(CORPUS)
    a = evaluate(NoisyBackend(error_rate=0.25, seed=3), corpus)
    b = evaluate(NoisyBackend(error_rate=0.25, seed=3), corpus)
    assert a.cer == b.cer and a.per_hand == b.per_hand


def test_gate_thresholds_are_the_policy_knob():
    report = evaluate(NoisyBackend(error_rate=0.05, seed=2), load_corpus(CORPUS))
    strict = gates_production(report, ScribeThresholds(max_cer=0.0, max_wer=0.0,
                                                       min_field_accuracy=1.0))
    lenient = gates_production(report, ScribeThresholds(max_cer=0.9, max_wer=0.9,
                                                        min_field_accuracy=0.0))
    assert not strict.passed and lenient.passed  # the same run, gated by the bar


def test_empty_corpus_is_blocked_not_silently_passed():
    report = evaluate(PerfectBackend(), [])
    gate = gates_production(report)
    assert not gate.passed and any("empty" in r for r in gate.reasons)


def test_per_hand_breakdown_surfaces_where_difficulty_hides():
    report = evaluate(NoisyBackend(error_rate=0.2, seed=4), load_corpus(CORPUS))
    assert "secretary" in report.per_hand and "print" in report.per_hand
