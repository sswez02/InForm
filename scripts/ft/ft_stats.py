from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import statistics


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    path = Path("data/ft/train_clean.jsonl")

    num_samples = 0
    output_lengths: List[int] = []
    query_lengths: List[int] = []
    context_counts: List[int] = []
    mode_counts: Dict[str, int] = {}

    for obj in iter_jsonl(path):
        num_samples += 1

        instr = obj.get("instruction") or ""
        input_obj: Dict[str, Any] = obj.get("input") or {}
        query = (input_obj.get("query") or "").strip()
        mode = (input_obj.get("mode") or "").strip() or "unknown"
        context = input_obj.get("context") or []
        output = (obj.get("output") or "").strip()

        output_lengths.append(len(output))
        query_lengths.append(len(query))
        context_counts.append(len(context))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    def mean(xs: List[int]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def median(xs: List[int]) -> float:
        return float(statistics.median(xs)) if xs else 0.0

    stats = {
        "path": str(path),
        "num_samples": num_samples,
        "output_chars_mean": mean(output_lengths),
        "output_chars_median": median(output_lengths),
        "query_chars_mean": mean(query_lengths),
        "query_chars_median": median(query_lengths),
        "context_passages_mean": mean(context_counts),
        "context_passages_median": median(context_counts),
        "mode_counts": mode_counts,
    }

    out_path = Path("data/ft/stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Wrote stats to {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
