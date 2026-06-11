"""Retrieval ablation eval.

Runs each configuration over the gold question set and reports recall@k, MRR@10,
and nDCG@10 -- overall and broken out by question category. The per-category
breakdown is the point: it shows graph expansion lifting *relational* recall
without degrading *lookup*/*provenance*, which is the central claim of the system.

  python eval/run_eval.py            # full ablation
  python eval/run_eval.py --fast     # skip the rerank configs (no cross-encoder)

Writes eval/results.md and eval/results.json. Deterministic given the seeded corpus.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genealogy_rag.corpus import Question, load_questions  # noqa: E402
from genealogy_rag.manifest import (  # noqa: E402
    build_manifest,
    corpus_digest,
    model_ref,
    results_digest,
)
from genealogy_rag.pipeline import GenealogyRAG  # noqa: E402
from genealogy_rag.retrieval import RetrievalConfig  # noqa: E402

K_VALUES = (1, 3, 5, 10)
MRR_CUTOFF = 10
NDCG_CUTOFF = 10

ABLATIONS = [
    ("vector-only", RetrievalConfig(use_vector=True, use_bm25=False,
                                    use_graph=False, use_rerank=False)),
    ("bm25-only", RetrievalConfig(use_vector=False, use_bm25=True,
                                  use_graph=False, use_rerank=False)),
    ("hybrid (vector+bm25)", RetrievalConfig(use_vector=True, use_bm25=True,
                                             use_graph=False, use_rerank=False)),
    ("hybrid + graph", RetrievalConfig(use_vector=True, use_bm25=True,
                                       use_graph=True, use_rerank=False)),
    ("hybrid + graph + rerank", RetrievalConfig(use_vector=True, use_bm25=True,
                                                use_graph=True, use_rerank=True)),
]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str], cutoff: int) -> float:
    for rank, did in enumerate(retrieved[:cutoff], 1):
        if did in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1)
              for i, did in enumerate(retrieved[:k], 1) if did in relevant)
    ideal = sum(1.0 / math.log2(i + 1)
                for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(rag: GenealogyRAG, questions: list[Question],
             cfg: RetrievalConfig) -> dict:
    agg: dict[str, list[float]] = defaultdict(list)
    per_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q in questions:
        rel = set(q.relevant_doc_ids)
        hits = rag.retrieve(q.question, k=max(K_VALUES), cfg=cfg)
        retrieved = [h.doc.id for h in hits]
        for k in K_VALUES:
            r = recall_at_k(retrieved, rel, k)
            agg[f"recall@{k}"].append(r)
            per_cat[q.category][f"recall@{k}"].append(r)
        mrr = reciprocal_rank(retrieved, rel, MRR_CUTOFF)
        ndcg = ndcg_at_k(retrieved, rel, NDCG_CUTOFF)
        agg["mrr@10"].append(mrr)
        agg["ndcg@10"].append(ndcg)
        per_cat[q.category]["mrr@10"].append(mrr)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "overall": {m: round(mean(v), 3) for m, v in agg.items()},
        "by_category": {c: {m: round(mean(v), 3) for m, v in d.items()}
                        for c, d in per_cat.items()},
    }


def fmt_overall_table(results: list[tuple[str, dict]]) -> str:
    head = "| configuration | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 | nDCG@10 |"
    sep = "|---|---|---|---|---|---|---|"
    rows = [head, sep]
    for name, res in results:
        o = res["overall"]
        rows.append(
            f"| {name} | {o['recall@1']:.3f} | {o['recall@3']:.3f} | "
            f"{o['recall@5']:.3f} | {o['recall@10']:.3f} | {o['mrr@10']:.3f} | "
            f"{o['ndcg@10']:.3f} |")
    return "\n".join(rows)


def fmt_category_table(results: list[tuple[str, dict]], cats: list[str]) -> str:
    head = "| configuration | " + " | ".join(f"{c} R@5" for c in cats) + " |"
    sep = "|---|" + "|".join("---" for _ in cats) + "|"
    rows = [head, sep]
    for name, res in results:
        cells = []
        for c in cats:
            v = res["by_category"].get(c, {}).get("recall@5")
            cells.append(f"{v:.3f}" if v is not None else "—")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip rerank configs (no cross-encoder download)")
    args = ap.parse_args()

    t0 = time.time()
    questions = load_questions()
    rag = GenealogyRAG()
    print(f"corpus: {len(rag.documents)} docs | questions: {len(questions)} | "
          f"vector backend: {rag.retriever.backend_vector}", file=sys.stderr)

    ablations = [a for a in ABLATIONS
                 if not (args.fast and a[1].use_rerank)]
    results: list[tuple[str, dict]] = []
    for name, cfg in ablations:
        ts = time.time()
        res = evaluate(rag, questions, cfg)
        results.append((name, res))
        print(f"  [{name}] R@5={res['overall']['recall@5']:.3f} "
              f"MRR={res['overall']['mrr@10']:.3f} ({time.time()-ts:.1f}s)",
              file=sys.stderr)

    cats = ["lookup", "provenance", "household", "relational"]
    overall_tbl = fmt_overall_table(results)
    cat_tbl = fmt_category_table(results, cats)

    md = [
        "# Retrieval evaluation\n",
        f"Corpus: {len(rag.documents)} source documents · {len(questions)} gold "
        f"questions · embedding: `all-MiniLM-L6-v2` · vector backend: "
        f"`{rag.retriever.backend_vector}`.\n",
        "## Overall (ablation)\n", overall_tbl, "",
        "## Recall@5 by question category\n", cat_tbl, "",
        "_Generated by `eval/run_eval.py`; deterministic given the seeded corpus._",
    ]
    out_md = Path(__file__).resolve().parent / "results.md"
    out_md.write_text("\n".join(md) + "\n")
    results_obj = {name: res for name, res in results}
    (out_md.with_suffix(".json")).write_text(json.dumps(results_obj, indent=2))

    # Attested run manifest: bind the weights that ran to the corpus and the results, so a
    # published number can be re-derived and checked. The embedder was loaded for the eval;
    # the reranker is attested only when a rerank config actually ran (else it isn't loaded,
    # and we don't download a model just to fingerprint it).
    models = [model_ref("embedder", rag.retriever.embedder.attest())]
    if any(cfg.use_rerank for _, cfg in ablations):
        models.append(model_ref("reranker", rag.retriever.reranker.attest()))
    manifest = build_manifest(
        models,
        corpus=corpus_digest((d.id, d.searchable) for d in rag.documents),
        corpus_docs=len(rag.documents),
        config={"ablations": [name for name, _ in ablations], "k_values": list(K_VALUES),
                "questions": len(questions)},
        results=results_digest(results_obj))
    (out_md.parent / "manifest.json").write_text(manifest.to_json() + "\n")

    print("\n" + overall_tbl + "\n\n" + cat_tbl)
    print(f"\nattested run {manifest.manifest_id} "
          f"({len(models)} model(s), corpus {manifest.corpus}, results {manifest.results})")
    print(f"wrote {out_md}, results.json, and manifest.json  (total {time.time()-t0:.1f}s)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
