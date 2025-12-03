from __future__ import annotations

import argparse
from pathlib import Path

from src.store import StudyStore
from src.dense_retriever import DenseRetriever
from src.answerer import answer_query, Mode


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
        help="How many passages to retrieve before composing",
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=3,
        help="Maximum number of distinct studies to cite",
    )
    args = parser.parse_args()
    mode: Mode = args.mode
    query = args.query
    expanded_query = query

    store = StudyStore.from_dir(Path("data/studies"))
    studies = store.get_all_studies()
    passages = store.get_all_passages()

    retriever = DenseRetriever()
    retriever.add_passages(passages)

    ans = answer_query(
        mode=mode,
        query=args.query,
        retriever=retriever,
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


if __name__ == "__main__":
    main()
