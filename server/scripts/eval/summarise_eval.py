from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

import matplotlib.pyplot as plt


EVAL_PATH = Path("data/eval/batch_eval.json")
SUMMARY_MD_PATH = Path("data/eval/eval_summary.md")


def load_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of records in batch_eval.json")
    return data


def main() -> None:
    results = load_results(EVAL_PATH)

    # Filter to successful LLM responses
    records: List[Dict[str, Any]] = []
    for r in results:
        if r.get("backend") != "llm":
            continue
        if r.get("ok") is False:
            continue  # skip failed calls
        if "answer" not in r and "response" not in r:
            continue

        # Normalise shape
        if "response" in r and "answer" not in r:
            resp = r["response"]
            records.append(
                {
                    "query": r.get("query"),
                    "mode": r.get("mode"),
                    "backend": "llm",
                    "answer": resp.get("answer", ""),
                    "answer_length": len(resp.get("answer", "")),
                    "num_citations": len(resp.get("citations", [])),
                    "confidence": resp.get("confidence", {}),
                }
            )
        else:
            records.append(r)

    if not records:
        print("No valid LLM records found in batch_eval.json")
        return

    # Stats
    total_answers = len(records)
    answer_lengths: List[int] = []
    citations_counts: List[int] = []

    conf_labels_overall: Counter[str] = Counter()
    answer_lengths_by_mode: Dict[str, List[int]] = {"beginner": [], "intermediate": []}
    citations_by_mode: Dict[str, List[int]] = {"beginner": [], "intermediate": []}
    conf_labels_by_mode: Dict[str, Counter[str]] = {
        "beginner": Counter(),
        "intermediate": Counter(),
    }

    for r in records:
        mode = r.get("mode", "beginner")
        ans_len = int(r.get("answer_length", len(r.get("answer", ""))))
        num_cits = int(r.get("num_citations", 0))

        answer_lengths.append(ans_len)
        citations_counts.append(num_cits)

        if mode in answer_lengths_by_mode:
            answer_lengths_by_mode[mode].append(ans_len)
            citations_by_mode[mode].append(num_cits)

        conf = r.get("confidence", {}) or {}
        label = (conf.get("label") or "").lower()
        if label in ("low", "medium", "high"):
            conf_labels_overall[label] += 1
            if mode in conf_labels_by_mode:
                conf_labels_by_mode[mode][label] += 1
        else:
            conf_labels_overall["unknown"] += 1
            if mode in conf_labels_by_mode:
                conf_labels_by_mode[mode]["unknown"] += 1

    def avg(xs: List[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    overall_avg_len = avg(answer_lengths)
    overall_avg_cits = avg(citations_counts)

    beginner_avg_len = avg(answer_lengths_by_mode["beginner"])
    inter_avg_len = avg(answer_lengths_by_mode["intermediate"])

    beginner_avg_cits = avg(citations_by_mode["beginner"])
    inter_avg_cits = avg(citations_by_mode["intermediate"])

    print("\n=== Batch Evaluation Summary (LLM-only) ===")
    print(f"Total answers: {total_answers}")
    print(f"Overall average answer length: {overall_avg_len:.1f} characters")
    print(f"Overall average citations per answer: {overall_avg_cits:.2f}\n")

    print("By mode:")
    print(
        f"  Beginner    - avg length: {beginner_avg_len:.1f}, avg citations: {beginner_avg_cits:.2f}"
    )
    print(
        f"  Intermediate - avg length: {inter_avg_len:.1f}, avg citations: {inter_avg_cits:.2f}\n"
    )

    print("Confidence distribution (overall):")
    total_conf = sum(conf_labels_overall.values())
    for label in ("low", "medium", "high", "unknown"):
        count = conf_labels_overall.get(label, 0)
        pct = 100.0 * count / total_conf if total_conf > 0 else 0.0
        print(f"  {label:8s}: {count:3d} ({pct:5.1f}%)")

    # Tables
    SUMMARY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_MD_PATH.open("w", encoding="utf-8") as f:
        f.write("# Batch Evaluation Summary (LLM-only)\n\n")
        f.write(f"- Total answers: **{total_answers}**\n")
        f.write(
            f"- Overall average answer length: **{overall_avg_len:.1f}** characters\n"
        )
        f.write(
            f"- Overall average citations per answer: **{overall_avg_cits:.2f}**\n\n"
        )

        f.write("## Average Answer Length and Citations by Mode\n\n")
        f.write("| Mode | Avg Answer Length (chars) | Avg Citations per Answer |\n")
        f.write("|------|---------------------------|---------------------------|\n")
        f.write(
            f"| Beginner     | {beginner_avg_len:.1f} | {beginner_avg_cits:.2f} |\n"
        )
        f.write(f"| Intermediate | {inter_avg_len:.1f} | {inter_avg_cits:.2f} |\n\n")

        f.write("## Confidence Distribution (Overall)\n\n")
        f.write("| Confidence | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        for label in ("low", "medium", "high", "unknown"):
            count = conf_labels_overall.get(label, 0)
            pct = 100.0 * count / total_conf if total_conf > 0 else 0.0
            f.write(f"| {label.capitalize():10s} | {count:5d} | {pct:9.1f}% |\n")

    print(f"\nWrote Markdown summary to {SUMMARY_MD_PATH}")

    # Plots
    plots_dir = Path("data/eval/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Answer length histogram by mode
    plt.figure()
    bins = 20
    if answer_lengths_by_mode["beginner"]:
        plt.hist(
            answer_lengths_by_mode["beginner"],
            bins=bins,
            alpha=0.6,
            label="Beginner",
            color="tab:blue",
        )
    if answer_lengths_by_mode["intermediate"]:
        plt.hist(
            answer_lengths_by_mode["intermediate"],
            bins=bins,
            alpha=0.6,
            label="Intermediate",
            color="tab:orange",
        )
    plt.xlabel("Answer length (characters)")
    plt.ylabel("Count")
    plt.title("Answer Length Distribution by Mode (LLM)")
    plt.legend()
    out1 = plots_dir / "answer_length_hist_by_mode.png"
    plt.tight_layout()
    plt.savefig(out1)
    plt.close()
    print(f"Saved {out1}")

    # 2) Citations per answer histogram by mode
    if citations_counts:
        max_cits = max(citations_counts)
        bins_range = range(0, max_cits + 2)

        plt.figure()
        if citations_by_mode["beginner"]:
            plt.hist(
                citations_by_mode["beginner"],
                bins=bins_range,
                alpha=0.6,
                align="left",
                label="Beginner",
                color="tab:blue",
            )
        if citations_by_mode["intermediate"]:
            plt.hist(
                citations_by_mode["intermediate"],
                bins=bins_range,
                alpha=0.6,
                align="left",
                label="Intermediate",
                color="tab:orange",
            )
        plt.xlabel("Number of citations per answer")
        plt.ylabel("Count")
        plt.title("Citations per Answer by Mode (LLM)")
        plt.xticks(list(bins_range))
        plt.legend()
        out2 = plots_dir / "citations_hist_by_mode.png"
        plt.tight_layout()
        plt.savefig(out2)
        plt.close()
        print(f"Saved {out2}")
    else:
        print("No citation counts found to plot.")

    # 3) Confidence distribution by mode (grouped bar chart)
    conf_labels = ["low", "medium", "high"]
    x = range(len(conf_labels))

    beginner_conf = [conf_labels_by_mode["beginner"].get(lbl, 0) for lbl in conf_labels]
    inter_conf = [
        conf_labels_by_mode["intermediate"].get(lbl, 0) for lbl in conf_labels
    ]

    if any(beginner_conf) or any(inter_conf):
        width = 0.35
        plt.figure()
        x_vals = list(x)
        plt.bar(
            [xi - width / 2 for xi in x_vals],
            beginner_conf,
            width=width,
            label="Beginner",
            color="tab:blue",
        )
        plt.bar(
            [xi + width / 2 for xi in x_vals],
            inter_conf,
            width=width,
            label="Intermediate",
            color="tab:orange",
        )
        plt.xticks(x_vals, [lbl.capitalize() for lbl in conf_labels])
        plt.xlabel("Confidence label")
        plt.ylabel("Count")
        plt.title("Confidence Distribution by Mode (LLM)")
        plt.legend()
        out3 = plots_dir / "confidence_distribution_by_mode.png"
        plt.tight_layout()
        plt.savefig(out3)
        plt.close()
        print(f"Saved {out3}")
    else:
        print("No confidence labels found to plot by mode.")


if __name__ == "__main__":
    main()
