from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

from src.store import StudyStore
from src.models import Study, Passage
from src.hybrid_retriever import HybridRetriever
from src.retriever import Retriever
from src.answerer import answer_query, Mode
from src.ft.generation import DomainLLM
from src.ft.formatting import build_prompt
from scripts.eval_report import load_test_queries


def make_retriever(passages: List[Passage]) -> Retriever:
    # Retriever weights can be adjusted
    retriever = HybridRetriever(tfidf_weight=0.4, dense_weight=0.6)
    retriever.add_passages(passages)
    return retriever


def select_context_for_llm(
    retrieval_results: List[Tuple[Passage, float]],
    max_passages: int = 8,
) -> List[Dict[str, Any]]:
    ctx: List[Dict[str, Any]] = []
    for rank, (p, _score) in enumerate(retrieval_results[:max_passages], start=1):
        ctx.append(
            {
                "study_id": p.study_id,
                "citation_index": rank,
                "section": p.section,
                "text": p.text,
            }
        )
    return ctx


def simple_overlap_score(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


def count_citation_tokens(text: str) -> Dict[str, Any]:
    """
    "count": 4,          # total number of citation tags
    "unique": [1, 2, 3]  # which citation numbers appear
    """
    matches = re.findall(r"\[(\d+)\]", text)
    ints = [int(m) for m in matches] if matches else []
    unique = sorted(set(ints))
    return {
        "count": len(ints),
        "unique": unique,
    }


def eval_finetuned_llm() -> Dict[str, Any]:
    studies_dir = Path("data/studies")
    test_path = Path("data/eval/test_queries.json")
    out_path = Path("data/eval/llm_vs_baseline.json")

    store = StudyStore.from_dir(studies_dir)
    studies = store.get_all_studies()
    passages = store.get_all_passages()
    studies_by_id = {s.id: s for s in studies}

    retriever = make_retriever(passages)
    test = load_test_queries(test_path)

    # Load fine-tuned LLM
    llm = DomainLLM(
        base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        lora_dir="model_lora",
        max_new_tokens=512,
    )

    per_query_results: List[Dict[str, Any]] = []

    for item in test:
        query = item["query"]
        mode_str = item.get("mode", "beginner")
        mode: Mode = mode_str

        print(f"\n=== Evaluating query: {query!r} (mode={mode}) ===")

        # Retrieve context
        retrieval_results = retriever.search(query, top_k=12)
        context_for_llm = select_context_for_llm(retrieval_results, max_passages=8)

        # Baseline answer
        baseline_answer = answer_query(
            mode=mode,
            query=query,
            retriever=retriever,
            studies=studies,
            top_k_passages=10,
            max_studies=3,
        )

        baseline_text = baseline_answer.answer_text.strip()
        baseline_citations = baseline_answer.references

        # LLM answer using same context
        instruction = (
            "Answer the user's question using the provided study excerpts. "
            "Include inline numeric citations like [1], [2] that correspond "
            "to the studies in the context."
        )
        llm_text = llm.generate_answer(
            instruction=instruction,
            query=query,
            context_passages=context_for_llm,
        )

        # Basic metrics
        overlap = simple_overlap_score(baseline_text, llm_text)
        base_len = len(baseline_text)
        llm_len = len(llm_text)
        len_ratio = llm_len / base_len if base_len > 0 else 0.0

        base_cits = count_citation_tokens(baseline_text)
        llm_cits = count_citation_tokens(llm_text)

        per_query_results.append(
            {
                "query": query,
                "mode": mode_str,
                "baseline": {
                    "text": baseline_text,
                    "length": base_len,
                    "citations": baseline_citations,
                    "citation_tokens": base_cits,
                },
                "llm": {
                    "text": llm_text,
                    "length": llm_len,
                    "citation_tokens": llm_cits,
                },
                "metrics": {
                    "overlap_jaccard": overlap,
                    "length_ratio": len_ratio,
                },
            }
        )

    # Average simple stats
    if per_query_results:
        avg_overlap = sum(
            r["metrics"]["overlap_jaccard"] for r in per_query_results
        ) / len(per_query_results)
        avg_len_ratio = sum(
            r["metrics"]["length_ratio"] for r in per_query_results
        ) / len(per_query_results)
    else:
        avg_overlap = 0.0
        avg_len_ratio = 0.0

    summary = {
        "num_queries": len(per_query_results),
        "avg_overlap_jaccard": avg_overlap,
        "avg_length_ratio": avg_len_ratio,
    }

    report = {
        "summary": summary,
        "per_query": per_query_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== LLM vs baseline summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote detailed report to {out_path}")

    return report


def main() -> None:
    eval_finetuned_llm()


if __name__ == "__main__":
    main()
