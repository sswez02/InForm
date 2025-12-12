from datetime import datetime
from typing import Any, List, Tuple, Dict

from src.core.models import Passage

CURRENT_YEAR = datetime.now().year


def get_passage_year(p: Passage, study_years: Dict[int, int]) -> int | None:
    """
    Try to extract a publication year for a passage.

    Priority:
    1) p.year
    2) p.metadata['year' / 'pub_year' / 'publication_year']
    3) fallback from STUDY_YEAR_BY_ID using p.study_id
    """
    year = getattr(p, "year", None)
    if isinstance(year, int):
        return year

    meta = getattr(p, "metadata", None)
    if isinstance(meta, dict):
        y = meta.get("year") or meta.get("pub_year") or meta.get("publication_year")
        if isinstance(y, int):
            return y
        if isinstance(y, str) and y.isdigit():
            return int(y)

    # fallback: study-level year
    sid = getattr(p, "study_id", None)
    if isinstance(sid, int):
        return study_years.get(sid)

    return None


def rerank_by_recency(
    results: List[Tuple[Passage, float]],
    study_years: Dict[int, int],
    window_years: int = 5,
) -> List[Tuple[Passage, float]]:
    """
    Boost scores for studies published within the last `window_years`.

    - Strong boost for last 5 years
    - Smaller boost for 5-10 years
    - Mild penalty for very old work
    """
    if not results:
        return results

    cutoff_recent = CURRENT_YEAR - window_years  # e.g. 2020 if now is 2025
    cutoff_moderate = CURRENT_YEAR - 10  # e.g. 2015

    boosted: List[Tuple[Passage, float]] = []
    for p, score in results:
        year = get_passage_year(p, study_years)
        bonus = 0.0

        if year is not None:
            if year >= cutoff_recent:
                bonus = 0.25  # big bump for last 5y
            elif year >= cutoff_moderate:
                bonus = 0.12  # medium bump for 5-10y
            elif year <= CURRENT_YEAR - 20:
                bonus = -0.10  # small penalty for >20y old (e.g. 2002)

        boosted.append((p, score + bonus))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted
