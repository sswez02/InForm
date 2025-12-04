from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


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
    path = Path("data/eval/human_eval_template.jsonl")
    if not path.exists():
        raise FileNotFoundError(path)

    clarity_base: List[float] = []
    clarity_llm: List[float] = []
    factual_base: List[float] = []
    factual_llm: List[float] = []
    cit_base: List[float] = []
    cit_llm: List[float] = []

    pref_counts: Dict[str, int] = {"baseline": 0, "llm": 0, "tie": 0}

    for obj in iter_jsonl(path):
        cb = obj.get("clarity_baseline")
        cl = obj.get("clarity_llm")
        fb = obj.get("factuality_baseline")
        fl = obj.get("factuality_llm")
        cfb = obj.get("citation_faithfulness_baseline")
        cfl = obj.get("citation_faithfulness_llm")
        pref = obj.get("overall_preference")

        if cb is not None and cl is not None:
            clarity_base.append(float(cb))
            clarity_llm.append(float(cl))
        if fb is not None and fl is not None:
            factual_base.append(float(fb))
            factual_llm.append(float(fl))
        if cfb is not None and cfl is not None:
            cit_base.append(float(cfb))
            cit_llm.append(float(cfl))
        if pref in pref_counts:
            pref_counts[pref] += 1

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "num_rated": len(clarity_base),
        "clarity": {
            "baseline": avg(clarity_base),
            "llm": avg(clarity_llm),
        },
        "factuality": {
            "baseline": avg(factual_base),
            "llm": avg(factual_llm),
        },
        "citation_faithfulness": {
            "baseline": avg(cit_base),
            "llm": avg(cit_llm),
        },
        "overall_preference_counts": pref_counts,
    }

    out_path = Path("data/eval/human_eval_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Human evaluation summary:")
    print(json.dumps(summary, indent=2))
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
