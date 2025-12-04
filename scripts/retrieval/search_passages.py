from pathlib import Path

from src.core.store import StudyStore
from src.retrieval.indexer import TfIdfIndex


def main() -> None:
    store = StudyStore.from_dir(Path("data/studies"))
    studies = store.get_all_studies()
    passages = store.get_all_passages()

    print(f"Loaded {len(studies)} studies, {len(passages)} passages.")

    index = TfIdfIndex()
    index.add_passages(passages)
    index.build()
    print(f"Index built with vocab size {len(index.vocab)}.")

    # Build study lookup with study_id -> study
    study_lookup = {s.id: s for s in studies}

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
        results = index.search(query, top_k=5)
        if not results:
            print("No results found.")
            continue

        # Output
        print(f"\nTop results for: {query!r}")
        for p, score in results:
            study = study_lookup.get(p.study_id)
            title = study.title if study else f"Study {p.study_id}"
            print("-" * 80)
            print(f"Score: {score:.3f}")
            print(f"Study: {title} (id={p.study_id}, section={p.section})")
            print(p.text[:400] + ("..." if len(p.text) > 400 else ""))
        print("-" * 80)
        # Example output:
        # --------------------------------------------------------------------------------
        # Score: 0.842
        # Study: Creatine increases strength in trained men (id=1, section=abstract)
        # Creatine supplementation increased 1RM squat and bench press after 8 weeks...
        # --------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
