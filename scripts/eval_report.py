from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.store import StudyStore
from src.indexer import TfIdfIndex
from src.retriever import Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.text_utils import tokenise
from src.models import Study, Passage


def load_test_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_recall_mrr_for_query(
    results: List[Tuple[Passage, float]],
    relevant_studies: List[int],
    k_values: List[int],
) -> Dict[str, float]:
    rel_set = set(relevant_studies)
    metrics: Dict[str, float] = {}
    retrieved_studies = [p.study_id for (p, _score) in results]

    for k in k_values:
        top_k = retrieved_studies[:k]

        # recall@k
        hit = any(s in rel_set for s in top_k)
        metrics[f"recall@{k}"] = 1.0 if hit else 0.0

        # mrr@k
        rr = 0.0
        for idx, sid in enumerate(top_k):
            if sid in rel_set:
                rr = 1.0 / (idx + 1)
                break
        metrics[f"mrr@{k}"] = rr

    return metrics


def top1_alignment(
    top_passage: Passage | None,
    studies_by_id: Dict[int, Study],
    target_training_status: str | None,
    target_outcomes: List[str] | None,
) -> Dict[str, float]:
    """
    Returns:
      - training_status_match: 1/0
      - outcome_match: 1/0
    """

    # Missing study or top passage
    if top_passage is None:
        return {"training_status_match": 0.0, "outcome_match": 0.0}

    study = studies_by_id.get(top_passage.study_id)
    if study is None:
        return {"training_status_match": 0.0, "outcome_match": 0.0}

    # Training status
    ts_match = 0.0
    if target_training_status:
        if study.training_status == target_training_status:
            ts_match = 1.0

    # Outcomes
    outcome_match = 0.0
    if target_outcomes:
        target_set = set(target_outcomes)
        primary = set(study.outcomes.get("primary", []))
        if primary & target_set:
            outcome_match = 1.0

    return {
        "training_status_match": ts_match,
        "outcome_match": outcome_match,
    }


def make_retriever(
    kind: str,
    passages: list[Passage],
    tfidf_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> Retriever:
    if kind == "tfidf":
        r = TfIdfIndex()
        r.add_passages(passages)
        r.build()
        return r
    elif kind == "dense":
        r = DenseRetriever()
        r.add_passages(passages)
        return r
    elif kind == "hybrid":
        r = HybridRetriever(tfidf_weight=tfidf_weight, dense_weight=dense_weight)
        r.add_passages(passages)
        return r
    else:
        raise ValueError(f"Unknown retriever kind: {kind}")


def eval_report(
    retriever: Retriever,
    studies_by_id: Dict[int, Study],
    k_values: List[int],
    test_queries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    per_query: List[Dict[str, Any]] = []
    sums: Dict[str, float] = {}
    count = 0

    for item in test_queries:
        query = item["query"]
        relevant = item.get("relevant_studies", [])
        target_ts = item.get("target_training_status")
        target_outcomes = item.get("target_outcomes", [])

        results = retriever.search(query, top_k=max(k_values))
        top_passage = results[0][0] if results else None

        metrics = compute_recall_mrr_for_query(results, relevant, k_values)
        align = top1_alignment(
            top_passage=top_passage,
            studies_by_id=studies_by_id,
            target_training_status=target_ts,
            target_outcomes=target_outcomes,
        )

        qrep: Dict[str, Any] = {
            "query": query,
            "relevant_studies": relevant,
            "metrics": metrics,
            "alignment": align,
            "top1_study_id": top_passage.study_id if top_passage else None,
        }
        per_query.append(qrep)

        # Sum up metrics
        count += 1
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        for k, v in align.items():
            sums[k] = sums.get(k, 0.0) + v

    # Average
    avg: Dict[str, float] = {}
    for k, v in sums.items():
        avg[k] = v / max(1, count)

    return {
        "num_queries": count,
        "avg_metrics": avg,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retriever backends")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["tfidf", "dense", "hybrid"],
        default="tfidf",
        help="Which retrieval backend to evaluate.",
    )
    parser.add_argument(
        "--tfidf-weight",
        type=float,
        default=0.5,
        help="Weight for TF-IDF in hybrid mode.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
        help="Weight for dense retriever in hybrid mode.",
    )
    args = parser.parse_args()

    studies_dir = Path("data/studies")
    test_path = Path("data/eval/test_queries.json")
    out_path = Path("data/eval/report.json")

    store = StudyStore.from_dir(studies_dir)
    studies = store.get_all_studies()
    passages = store.get_all_passages()
    studies_by_id = {s.id: s for s in studies}
    test = load_test_queries(test_path)

    # Build retriever using backend + weights
    retriever = make_retriever(
        kind=args.backend,
        passages=passages,
        tfidf_weight=args.tfidf_weight,
        dense_weight=args.dense_weight,
    )

    report = eval_report(
        retriever=retriever,
        studies_by_id=studies_by_id,
        k_values=[1, 3, 5],
        test_queries=test,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote eval report to {out_path}")
    print("Summary:")
    print(json.dumps(report["avg_metrics"], indent=2))


if __name__ == "__main__":
    main()
