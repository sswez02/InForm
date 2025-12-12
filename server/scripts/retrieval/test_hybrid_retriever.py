from __future__ import annotations

from pathlib import Path

from src.core.store import StudyStore
from src.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    studies_dir = Path("data/studies")
    store = StudyStore.from_dir(studies_dir)

    studies = store.get_all_studies()
    passages = store.get_all_passages()

    print(f"Loaded {len(passages)} passages.")

    retriever = HybridRetriever(tfidf_weight=0.5, dense_weight=0.5)
    retriever.add_passages(passages)

    print("Hybrid retriever initialised (TF-IDF + Dense)")

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
            print(
                f"Hybrid score: {score:.3f} | study_id={p.study_id}, section={p.section}"
            )
            print(p.text[:400] + ("..." if len(p.text) > 400 else ""))
        print("-" * 80)


if __name__ == "__main__":
    main()
