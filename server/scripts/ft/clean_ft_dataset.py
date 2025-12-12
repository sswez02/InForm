from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def iter_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
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
    in_path = Path("data/ft/train.jsonl")
    out_path = Path("data/ft/train_clean.jsonl")

    min_output_chars = 80  # drop short answers
    max_output_chars = 4000  # avoid long ones
    min_context_passages = 1  # require at least one context chunk

    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for obj in iter_jsonl(in_path):
            output = (obj.get("output") or "").strip()
            input_obj: Dict[str, Any] = obj.get("input") or {}
            context = input_obj.get("context") or []

            if len(output) < min_output_chars:
                dropped += 1
                continue
            if len(output) > max_output_chars:
                dropped += 1
                continue
            if len(context) < min_context_passages:
                dropped += 1
                continue

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept {kept} samples, dropped {dropped}")
    print(f"Wrote cleaned dataset to {out_path}")


if __name__ == "__main__":
    main()
