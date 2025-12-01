from __future__ import annotations

import argparse
from pathlib import Path

from src.load_studies import load_studies_from_dir
from src.indexer import TfIdfIndex
from src.answerer import answer_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the agent a question")
    # Required argument ( query )
    parser.add_argument("query", type=str, help="Your question / prompt")
    # Optional arguments ( top_k_passsages, max_studies )
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

    studies_dir = Path("data/studies")
    studies, passages = load_studies_from_dir(studies_dir)

    index = TfIdfIndex()
    index.add_passages(passages)
    index.build()

    ans = answer_query(
        query=args.query,
        index=index,
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
