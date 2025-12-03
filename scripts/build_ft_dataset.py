from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.store import StudyStore
from src.models import Passage, Study


def load_interactions(path: Path) -> List[Dict[str, Any]]:
    interactions: List[Dict[str, Any]] = []

    if not path.exists():
        print(f"Issue: interactions log {path} does not exist.")
        return interactions

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "interaction":
                interactions.append(obj)
    return interactions


def build_passage_lookup(passages: List[Passage]) -> Dict[int, Passage]:
    return {p.id: p for p in passages}


def make_instruction(mode: str) -> str:
    if mode == "beginner":
        return (
            "Answer the user's question in BEGINNER mode, using the provided study "
            "excerpts, explaining in simple, non-technical language, and include "
            "inline numeric citations like [1], [2] that map to the studies."
        )
    else:
        return (
            "Answer the user's question in INTERMEDIATE mode, using the provided "
            "study excerpts, including relevant methodological details where "
            "helpful, and include inline numeric citations like [1], [2] that "
            "map to the studies."
        )


def build_context(
    interaction: Dict[str, Any],
    passage_by_id: Dict[int, Passage],
) -> List[Dict[str, Any]]:
    ctx: List[Dict[str, Any]] = []

    retrieval = interaction.get("retrieval", {})
    results = retrieval.get("results", [])
    answer_refs = interaction.get("answer", {}).get("references", [])

    # Map study id to citation index
    study_to_citation_idx: Dict[int, int] = {}
    for ref in answer_refs:
        sid = ref.get("study_id")
        idx = ref.get("index")
        if sid is not None and idx is not None:
            study_to_citation_idx[sid] = idx
    # ctx contains study_id, citation_index, section, text
    for row in results:
        pid = row.get("passage_id")
        sid = row.get("study_id")
        if pid is None or sid is None:
            continue
        p = passage_by_id.get(pid)
        if p is None:
            continue
        citation_idx = study_to_citation_idx.get(sid)
        ctx.append(
            {
                "study_id": sid,
                "citation_index": citation_idx,
                "section": row.get("section", p.section),
                "text": p.text,
            }
        )

    return ctx


def main() -> None:
    interactions_path = Path("data/logs/interactions.jsonl")
    out_path = Path("data/ft/train.jsonl")
    store = StudyStore.from_dir(Path("data/studies"))

    passages = store.get_all_passages()
    passage_by_id = build_passage_lookup(passages)
    interactions = load_interactions(interactions_path)

    print(f"Loaded {len(interactions)} interactions.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_written = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for i in interactions:
            query = i.get("query", "")
            mode = i.get("mode", "beginner")
            answer = i.get("answer", {})
            answer_text = answer.get("text", "").strip()

            if not query or not answer_text:
                continue

            context = build_context(i, passage_by_id)

            if not context:
                continue

            # Full interaction record
            ft_sample = {
                "instruction": make_instruction(mode),
                "input": {
                    "query": query,
                    "mode": mode,
                    "context": context,
                },
                "output": answer_text,
            }

            f_out.write(json.dumps(ft_sample, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"Wrote {num_written} fine-tune samples to {out_path}")


if __name__ == "__main__":
    main()
