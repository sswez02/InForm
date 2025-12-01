from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from src.load_studies import load_studies_from_dir
from src.indexer import TfIdfIndex


def load_test_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_recall_and_mrr(
    index: TfIdfIndex,
    test_queries: List[Dict[str, Any]],
    k_values: List[int],
) -> Dict[str, float]:
    """
    Compute Recall@k and Mean Reciprocal Rank MRR@k over all test queries
    Returns a dict of metrics that can be printed
    """
    recalls = {k: 0 for k in k_values}
    mrrs = {k: 0.0 for k in k_values}
    n_queries = len(test_queries)

    for item in test_queries:
        query = item["query"]
        relevant_studies = set(item["relevant_studies"])

        # Perform a search for each test query
        max_k = max(k_values)
        results = index.search(query, top_k=max_k)
        retrieved_studies = [
            passage.study_id for passage, _ in results
        ]  # all search results

        # Evaluate if inside k range contains our decided relevant studies
        for k in k_values:
            topk_studies = retrieved_studies[:k]

            # Recall@k: did we get at least one relevant study in top-k
            topk_set = set(topk_studies)
            hit = len(topk_set & relevant_studies) > 0  # overlap
            if hit:
                recalls[k] += 1

            # MRR@k: 1/rank of first relevant in top-k, else 0
            rank = None
            for idx, study_id in enumerate(topk_studies):
                if study_id in relevant_studies:
                    rank = idx + 1  # ranks start at 1
                    break
            if rank is not None:
                mrrs[k] += 1.0 / rank  # relevance with early appearance

    metrics: Dict[str, float] = {}
    for k in sorted(k_values):
        metrics[f"recall@{k}"] = recalls[k] / max(1, n_queries)
        metrics[f"mrr@{k}"] = mrrs[k] / max(1, n_queries)
    return metrics


def main() -> None:
    studies_dir = Path("data/studies")
    test_path = Path("data/eval/test_queries.json")

    studies, passages = load_studies_from_dir(studies_dir)
    print(f"Loaded {len(studies)} studies, {len(passages)} passages.")

    index = TfIdfIndex()
    index.add_passages(passages)
    index.build()
    print(f"Index built with vocab size {len(index.vocab)}.")

    test_queries = load_test_queries(test_path)

    metrics = compute_recall_and_mrr(
        index=index,
        test_queries=test_queries,
        k_values=[1, 3, 5],
    )

    print(f"Evaluated {len(test_queries)} queries.")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
