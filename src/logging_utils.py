from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .models import Study, Passage
from .answerer import Answer


# Timestamp
def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_retrieval_log(
    query: str,
    expanded_query: str,
    results: List[Tuple[Passage, float]],
    studies_by_id: Dict[int, Study],
    top_k_passages: int,
) -> Dict[str, Any]:
    rows = []
    for rank, (p, score) in enumerate(results, start=1):
        s = studies_by_id[p.study_id]
        rows.append(
            {
                "rank": rank,
                "score": float(score),
                "passage_id": p.id,
                "study_id": p.study_id,
                "section": p.section,
                "training_status": getattr(s, "training_status", "unknown"),
                "rating": getattr(s, "rating", None),
                "year": getattr(s, "year", None),
            }
        )
        return {
            "top_k_passages": top_k_passages,
            "results": rows,
        }


def log_interaction(
    log_path: Path,
    query: str,
    mode: str,
    expanded_query: str,
    retrieval_log: Dict[str, Any],
    answer: Answer,
) -> None:
    entry = {
        "type": "interaction",
        "timestamp": iso_utc_now(),
        "query": query,
        "mode": mode,
        "expanded_query": expanded_query,
        "retrieval": retrieval_log,
        "answer": {
            "text": answer.answer_text,
            "confidence": getattr(answer, "confidence", None),
            "references": answer.references,
        },
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
