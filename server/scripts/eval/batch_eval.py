from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Any, Dict, List

import requests

API_URL = "http://127.0.0.1:8000/ask"

# 50 sample evaluation questions
TEST_QUERIES: List[str] = [
    # Creatine basics
    "Is creatine safe for beginners?",
    "How much creatine should I take per day?",
    "Do I need a creatine loading phase?",
    "Does creatine cause water retention or bloating?",
    "Is creatine safe for long-term use?",
    "Can teenagers take creatine safely?",
    "Should I take creatine before or after my workout?",
    "Does creatine help with endurance or just strength?",
    "Is creatine monohydrate better than other forms of creatine?",
    "Do I have to cycle off creatine?",
    # Protein & nutrition
    "How much protein do I need per day to build muscle?",
    "Is whey protein better than whole food protein for muscle growth?",
    "Does protein timing around the workout really matter?",
    "Can I build muscle on a calorie deficit?",
    "Does casein protein before bed improve muscle gain?",
    # Training frequency & splits
    "Is full-body training better than a bro split for muscle growth?",
    "How many times per week should I train each muscle group?",
    "Is push–pull–legs a good split for intermediates?",
    "Can I make progress training each muscle only once per week?",
    "Is it okay to train with sore muscles?",
    # Volume & intensity
    "How many sets per week should I do for hypertrophy?",
    "Is training to failure necessary for muscle growth?",
    "What rep ranges are best for hypertrophy?",
    "Is high volume better than low volume for strength?",
    "How much rest should I take between sets for hypertrophy?",
    # Exercise selection & technique
    "Are compound lifts enough to build all major muscle groups?",
    "Are machines as effective as free weights for building muscle?",
    "Does squat depth affect hypertrophy and strength gains?",
    "Are deadlifts necessary for building a strong back?",
    "Is it better to do barbell bench press or dumbbell bench press?",
    # Periodisation & progression
    "What is progressive overload and how fast should I add weight?",
    "Is linear periodisation effective for intermediates?",
    "Do deload weeks improve strength and hypertrophy long term?",
    "Should I change my routine frequently to avoid plateaus?",
    "Is daily undulating periodisation better than linear periodisation?",
    # Special populations / context
    "Can women build muscle as effectively as men with resistance training?",
    "Is resistance training safe for older adults?",
    "Does creatine help older adults maintain muscle mass?",
    "Is high-intensity resistance training safe for beginners?",
    "Does training in a fasted state affect muscle growth?",
    # Recovery & fatigue
    "How much sleep do I need for optimal muscle growth?",
    "Does training to failure increase injury risk?",
    "How many rest days per week should I take?",
    "Does active recovery speed up muscle recovery?",
    "How does stress affect strength and hypertrophy?",
    # Hypertrophy vs strength
    "Can I train for strength and hypertrophy at the same time?",
    "Is low-rep high-weight training better for strength?",
    "Can I gain strength without gaining much muscle mass?",
    "Are strength gains mostly neural at the beginning of training?",
    "What is the minimum effective dose for hypertrophy?",
]


def call_agent(
    query: str,
    mode: Literal["beginner", "intermediate"],
    timeout: float = 20.0,
) -> Dict[str, Any]:
    """
    Call the FastAPI /ask endpoint using the LLM backend
    """
    payload = {
        "mode": mode,
        "query": query,
        "use_llm": True,
        "top_k_passages": 10,
        "max_studies": 3,
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=timeout)
    except Exception as e:
        return {
            "ok": False,
            "error": f"Request error: {e!r}",
            "mode": mode,
        }

    if not resp.ok:
        return {
            "ok": False,
            "status_code": resp.status_code,
            "error": resp.text,
            "mode": mode,
        }

    try:
        data = resp.json()
    except Exception as e:
        return {
            "ok": False,
            "error": f"JSON decode error: {e!r}",
            "raw_text": resp.text,
            "mode": mode,
        }

    return {
        "ok": True,
        "response": data,
        "mode": mode,
    }


def main() -> None:
    """
    Run 50 queries across two modes (beginner + intermediate)
    using the LLM backend, and export results to JSON
    """
    results: List[Dict[str, Any]] = []
    modes: List[Literal["beginner", "intermediate"]] = ["beginner", "intermediate"]

    for q_idx, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n=== Query {q_idx}/{len(TEST_QUERIES)}: {query!r} ===")

        for mode in modes:
            print(f"  -> mode={mode}, backend=llm ... ", end="", flush=True)
            result = call_agent(query=query, mode=mode)

            if result.get("ok") and "response" in result:
                resp = result["response"]
                answer = resp.get("answer", "")
                citations = resp.get("citations", [])
                confidence = resp.get("confidence", {})

                results.append(
                    {
                        "query": query,
                        "mode": mode,
                        "backend": "llm",
                        "answer": answer,
                        "answer_length": len(answer),
                        "num_citations": len(citations),
                        "citations": citations,
                        "studies": resp.get("studies", []),
                        "confidence": confidence,
                    }
                )
                print(f"OK (len={len(answer)}, cits={len(citations)})")
            else:
                results.append(
                    {
                        "query": query,
                        "mode": mode,
                        "backend": "llm",
                        "ok": False,
                        "error": result.get("error"),
                        "status_code": result.get("status_code"),
                    }
                )
                print("FAIL")

    out_path = Path("data/eval/batch_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
