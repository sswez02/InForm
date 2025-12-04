from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    in_path = Path("data/eval/llm_vs_baseline.json")
    out_path = Path("data/eval/human_eval_template.jsonl")

    if not in_path.exists():
        raise FileNotFoundError(f"Run eval_finetuned_llm.py first to create {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    per_query: List[Dict[str, Any]] = data.get("per_query", [])
    print(f"Loaded {len(per_query)} queries from {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        for item in per_query:
            obj = {
                "query": item["query"],
                "mode": item["mode"],
                "baseline_text": item["baseline"]["text"],
                "llm_text": item["llm"]["text"],
                "clarity_baseline": None,
                "clarity_llm": None,
                "factuality_baseline": None,
                "factuality_llm": None,
                "citation_faithfulness_baseline": None,
                "citation_faithfulness_llm": None,
                "overall_preference": None,  # "baseline" | "llm" | "tie"
            }
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote human evaluation template to {out_path}")


if __name__ == "__main__":
    main()
