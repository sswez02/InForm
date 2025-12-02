from __future__ import annotations

from pathlib import Path

from src.store import StudyStore
from src.dense_retriever import DenseRetriever


def main() -> None:
    store = StudyStore.from_dir(Path("data/studies"))
    passages = store.get_all_passages()

    print(f"Loaded {len(passages)} passages.")

    retriever = DenseRetriever()
    retriever.add_passages(passages)

    print("Dense embeddings built")

    while True:
        try:
            # Input
            query = input("\nQuery (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Quitting.")
            break

        # Search
        results = retriever.search(query, top_k=5)
        if not results:
            print("No results found.")
            continue

        # Output
        print(f"\nTop results for: {query!r}")
        for p, score in results:
            print("-" * 80)
            print(f"Score: {score:.3f}")
            print(f"study_id={p.study_id}, section={p.section})")
            print(p.text[:400] + ("..." if len(p.text) > 400 else ""))
        print("-" * 80)


if __name__ == "__main__":
    main()
