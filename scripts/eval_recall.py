from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from src.load_studies import load_studies_from_dir
from src.indexer import TfIdfIndex


def load_test_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def eval_recall_at_k(
    index: TfIdfIndex,
    test_queries: List[Dict[str, Any]],
    k_values: List[int],
) -> None:
    """
    For each k in k_values, compute recall@k over all test queries
    Recall@k = (number of queries with at least 1 hit in top k) / (total number of queries)
    """
    recalls = {k: 0 for k in k_values}
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
            topk_studies = set(retrieved_studies[:k])
            hit = len(topk_studies & relevant_studies) > 0  # overlap
            if hit:
                recalls[k] += 1

    print(f"Evaluated {n_queries} queries.")
    for k in sorted(k_values):
        r = recalls[k] / max(1, n_queries)
        print(f"Recall@{k}: {r:.3f}")


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

    eval_recall_at_k(
        index=index,
        test_queries=test_queries,
        k_values=[1, 3, 5],
    )


if __name__ == "__main__":
    main()
