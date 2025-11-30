from pathlib import Path
from src.load_studies import load_studies_from_dir


def main():
    studies_dir = Path("data/studies")
    studies, passages = load_studies_from_dir(studies_dir)

    print(f"Loaded {len(studies)} studies, {len(passages)} passages.")

    for s in studies:
        print(s)

    print("--- sample passages ---")
    for p in passages[:5]:
        print(p.section, ":", p.text[:80], "...")


if __name__ == "__main__":
    main()
