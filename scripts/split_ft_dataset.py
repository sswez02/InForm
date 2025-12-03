from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_all(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    in_path = Path("data/ft/train_clean.jsonl")
    out_dir = Path("data/ft/splits")

    items = load_all(in_path)
    print(f"Loaded {len(items)} cleaned samples")

    if not items:
        print("No data to split")
        return

    random.seed(42)
    random.shuffle(items)

    n = len(items)
    n_test = max(1, int(0.1 * n))
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val - n_test

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]

    print(f"Split into train={len(train)}, val={len(val)}, test={len(test)}")

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)

    print(f"Wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
