from __future__ import annotations
from typing import Any, Dict, List


def build_prompt(instruction: str, query: str, context_passages: list[dict]) -> str:
    # Prompt format
    context_segments = []
    for c in context_passages:
        idx = c.get("citation_index", None)
        idx_str = f"[{idx}]" if idx else ""
        section = c.get("section", "unknown")
        text = c.get("text", "")
        context_segments.append(f"{idx_str} ({section}) {text}")

    context_block = "\n".join(context_segments)

    prompt = (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Context:\n"
        f"{context_block}\n\n"
        "### Question:\n"
        f"{query}\n\n"
        "### Answer:\n"
    )

    return prompt
