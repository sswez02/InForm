from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.store import StudyStore
from src.core.models import Study, Passage
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import Retriever

from scripts.eval.eval_report import eval_report, load_test_queries


def run_for_weights(
    tfidf_w: float,
    dense_w: float,
    passages: List[Passage],
    studies_by_id: Dict[int, Study],
    test_queries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    retriever: Retriever = HybridRetriever(
        tfidf_weight=tfidf_w,
        dense_weight=dense_w,
    )
    retriever.add_passages(passages)
    report = eval_report(
        retriever=retriever,
        studies_by_id=studies_by_id,
        k_values=[1, 3, 5],
        test_queries=test_queries,
    )
    return report


def main() -> None:
    studies_dir = Path("data/studies")
    test_path = Path("data/eval/test_queries.json")
    out_path = Path("data/eval/hybrid_tuning.json")

    store = StudyStore.from_dir(studies_dir)
    studies = store.get_all_studies()
    passages = store.get_all_passages()
    studies_by_id = {s.id: s for s in studies}

    test = load_test_queries(test_path)

    # Tune weights
    weights = [
        (0.8, 0.2),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.4, 0.6),
        (0.2, 0.8),
    ]

    all_results: List[Dict[str, Any]] = []

    for tfidf_w, dense_w in weights:
        print(f"\n=== Evaluating tfidf={tfidf_w:.2f}, dense={dense_w:.2f} ===")
        report = run_for_weights(
            tfidf_w=tfidf_w,
            dense_w=dense_w,
            passages=passages,
            studies_by_id=studies_by_id,
            test_queries=test,
        )
        avg = report["avg_metrics"]
        print(json.dumps(avg, indent=2))
        all_results.append(
            {
                "tfidf_weight": tfidf_w,
                "dense_weight": dense_w,
                "avg_metrics": avg,
            }
        )

    out = {
        "results": all_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote hybrid tuning results to {out_path}")


if __name__ == "__main__":
    main()
