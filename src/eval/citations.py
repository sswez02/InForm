from __future__ import annotations

import re
from typing import Dict, List, Set


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citation_indexes(text: str) -> List[int]:
    """
    Return a list of all integer citation indexes found in text, e.g. [1], [2], [10]
    -> [1, 2, 10]
    """
    matches = CITATION_PATTERN.findall(text or "")
    return [int(m) for m in matches]


def check_citations(
    text: str,
    allowed_indexes: Set[int],
) -> Dict[str, float | List[int]]:
    """
    Check citations in text against a set of allowed indexes (e.g. {1,2,3}).

    Returns:
      {
        "total": total_citations,
        "unique": unique_citations,
        "valid": num_valid,
        "invalid": num_invalid,
        "valid_ratio": valid / max(total,1),
        "invalid_ratio": invalid / max(total,1),
        "hallucinated_indexes": [...],
        "used_indexes": [...],
      }
    """
    indexes = extract_citation_indexes(text)
    total = len(indexes)
    unique_indexes = sorted(set(indexes))

    valid = sum(1 for i in indexes if i in allowed_indexes)
    invalid = total - valid

    hallucinated = sorted({i for i in indexes if i not in allowed_indexes})

    return {
        "total": float(total),
        "unique": float(len(unique_indexes)),
        "valid": float(valid),
        "invalid": float(invalid),
        "valid_ratio": float(valid / total) if total > 0 else 0.0,
        "invalid_ratio": float(invalid / total) if total > 0 else 0.0,
        "hallucinated_indexes": hallucinated,
        "used_indexes": unique_indexes,
    }
