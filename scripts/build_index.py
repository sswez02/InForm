from pathlib import Path

from src.store import StudyStore
from src.indexer import TfIdfIndex


def main():
    studies_dir = Path("data/studies")
    store = StudyStore.from_dir(studies_dir)

    studies = store.get_all_studies()
    passages = store.get_all_passages()

    print(f"Loaded {len(studies)} studies, {len(passages)} passages.")

    index = TfIdfIndex()
    index.add_passages(passages)
    index.build()

    print(f"Index built. Vocab size: {len(index.vocab)}")

    # Testing
    for s in studies:
        print(s)
    print("--- sample passages ---")
    for p in passages[:5]:
        print(p.section, ":", p.text[:80], "...")


if __name__ == "__main__":
    main()
