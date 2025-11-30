from pathlib import Path

from src.load_studies import load_studies_from_dir
from src.indexer import TfIdfIndex


def main() -> None:
    studies_dir = Path("data/studies")
    studies, passages = load_studies_from_dir(studies_dir)

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
        for passage, score in results:
            study = study_lookup.get(passage.study_id)
            title = study.title if study else f"Study {passage.study_id}"
            print("-" * 80)
            print(f"Score: {score:.3f}")
            print(f"Study: {title} (id={passage.study_id}, section={passage.section})")
            print(passage.text[:400] + ("..." if len(passage.text) > 400 else ""))
        print("-" * 80)
        # Example output:
        # --------------------------------------------------------------------------------
        # Score: 0.842
        # Study: Creatine increases strength in trained men (id=1, section=abstract)
        # Creatine supplementation increased 1RM squat and bench press after 8 weeks...
        # --------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
