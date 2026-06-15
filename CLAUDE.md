# CLAUDE.md — genealogy-graphrag

Orientation for future Claude Code sessions. Keep this short and true.

## What this is

A hybrid retrieval system for **provenance-grounded genealogical question answering**.
It fuses dense (sentence-transformer) retrieval, sparse (BM25) retrieval, and structured
**kinship-graph resolution**, then reranks with a cross-encoder. Every answer carries the
bibliographic citations of the records it rests on.

The point it makes measurable: **dense and lexical retrieval cannot answer relational
questions** ("who was the maternal grandfather of X?") because the answer entity is never
named in the query — and a graph can. On the gold set, adding graph resolution takes
**relational recall@5 from 0.000 → 1.000** and overall **MRR@10 from 0.77 → 1.00**, with
no regression on the question types text retrieval already handles. Everything runs on CPU
with no external services; numbers come from `eval/run_eval.py`, not assertions.

## Pipeline

`query → relation-parse + entity-link → (dense MiniLM→FAISS-HNSW ∥ BM25) → RRF fusion`;
in parallel, when `<relation> of <person>` fires, the **kinship graph** (NetworkX / Neo4j)
resolves the target person and pins their identity records as authoritative. The fuzzy
(non-pinned) tail is cross-encoder reranked; pinned answers are merged first → ranked docs
+ citations.

## Layout

- `src/genealogy_rag/` — `config`, `corpus`, `embeddings`, `kinship`, `pipeline`,
  `provenance`, `rerank`, `retrieval`
- `src/genealogy_rag/graph/` — `networkx_store`, `neo4j_store`, `base`, `schema.cypher`
- `src/genealogy_rag/index/` — `lexical` (BM25), `vector` (FAISS)
- `src/genealogy_rag/` attestation (S0 work) — `attest`, `manifest`, `scribe`
- `data/genealogy/` — synthetic corpus (93 docs) + graph · `eval/` — 37 gold questions +
  ablation harness · `tests/`

## Commands

```bash
pip install -e ".[dev]"
pytest -q                 # note: test_pipeline.py needs sentence_transformers (CI dev extras)
python eval/run_eval.py   # reproduce the ablation table
make demo                 # S0 attestation --demo gate, if present
```

## Honest caveats (keep these honest)

- On this corpus, hybrid ≈ vector-only: named-entity questions are easy for dense
  retrieval and BM25's marginal contribution is small at this scale. **The headline lift
  is from the graph, not sparse+dense fusion.**
- Eval is on a synthetic 93-doc / 37-question gold set. Claims are scoped to it.

## Through-line

Part of **axiom-orion** — small, eval-driven pieces that turn a hand-waved claim into a
reproducible number; sibling to the **Vorion** (`@vorionsys`) governed-AI thesis. Built by
Ryan Cason.

## Notes / context (2026-06-15)

- The **`courier/art-director`** branch here was a *side project* (an unrelated design
  engine). It has been migrated to its own repo, **`axiom-orion/art-director`**, as a clean
  snapshot. That branch is now a redundant **stray** — safe to delete (proxy blocks branch
  deletion from web sessions).
- Other leftover branches from closed/merged PRs: `claude/publish-genealogy-graphrag-8c0ur`
  (PR #1, merged), `claude/flcason-agent-orchestration-9adb4y` (PR #5). The
  `claude/s0-weight-fingerprint` branch backs **open PR #6** — keep until that resolves.
