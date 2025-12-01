# scripts/ask.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.load_studies import load_studies_from_dir
from src.indexer import TfIdfIndex
from src.answerer import answer_query, Mode
from src.logging_utils import log_interaction, build_retrieval_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the agent a question")
    # Required argument ( query )
    parser.add_argument("query", type=str, help="Your question / prompt")
    # Optional arguments ( mode, top_k_passsages, max_studies )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["beginner", "intermediate"],
        default="beginner",
        help="Answer style.",
    )
    parser.add_argument(
        "--top-k-passages",
        type=int,
        default=10,
        help="How many passages to retrieve before composing.",
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=3,
        help="Maximum number of distinct studies to cite.",
    )
    args = parser.parse_args()
    mode: Mode = args.mode
    query = args.query

    studies_dir = Path("data/studies")
    log_path = Path("data/logs/interactions.jsonl")

    studies, passages = load_studies_from_dir(studies_dir)
    study_lookup = {s.id: s for s in studies}

    index = TfIdfIndex()
    index.add_passages(passages)
    index.build()

    raw_results = index.search(query, top_k=args.top_k_passages)

    ans = answer_query(
        mode=mode,
        query=args.query,
        retriever=index,
        studies=studies,
        top_k_passages=args.top_k_passages,
        max_studies=args.max_studies,
    )

    print("\n=== Answer ===")
    print(ans.answer_text)
    print("\n=== References ===")
    if not ans.references:
        print("(none)")
    else:
        for ref in ans.references:
            print(f"[{ref['index']}] {ref['citation']}")

    retrieval_log = build_retrieval_log(
        query=query,
        results=raw_results,
        studies_by_id=study_lookup,
        top_k_passages=args.top_k_passages,
    )

    log_interaction(
        log_path=log_path,
        query=query,
        mode=mode,
        retrieval_log=retrieval_log,
        answer=ans,
    )

    print("\n=== Confidence ===")
    print(ans.confidence.capitalize())


if __name__ == "__main__":
    main()
