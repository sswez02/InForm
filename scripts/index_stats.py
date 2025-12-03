from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.load_studies import load_studies_from_dir
from src.text_utils import tokenize


def main() -> None:
    studies_dir = Path("data/studies")
    # # Debug: see which JSON files are picked up
    # print("JSON files in data/studies:")
    # for path in sorted(studies_dir.glob("*.json")):
    #     print("  -", path.name)
    studies, passages = load_studies_from_dir(studies_dir)

    print(f"Studies: {len(studies)}")
    print(f"Passages: {len(passages)}")

    lengths = []
    all_tokens = Counter()

    for p in passages:
        tokens = tokenize(p.text)
        lengths.append(len(tokens))
        all_tokens.update(tokens)

    if lengths:
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
    else:
        avg_len = min_len = max_len = 0

    print(
        f"Passage length (tokens), avg: {avg_len:.1f} | min: {min_len}| max: {max_len}"
    )
    print("\nTop 20 tokens (excluding stopwords):")
    for token, count in all_tokens.most_common(20):
        print(f"  {token:15s} {count}")


if __name__ == "__main__":
    main()
