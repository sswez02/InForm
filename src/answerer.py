from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Any

from .models import Study, Passage
from .indexer import TfIdfIndex

Mode = Literal["beginner", "intermediate"]


@dataclass
class Answer:
    query: str
    answer_text: str
    references: List[Dict[str, Any]]  # {index, citation, study_id}


def _pick_studies_from_results(
    results: List[Tuple[Passage, float]], max_studies: int = 3
) -> List[Tuple[int, List[Passage]]]:
    """
    Group passages by study_id
    Returns list of (study_id, [passages_for_that_study]) sorted by best score
    """

    # Group passages and scores by study_id
    grouped: Dict[int, List[Tuple[Passage, float]]] = {}
    for passage, score in results:
        grouped.setdefault(passage.study_id, []).append((passage, score))

    # Compute an overall score for each study using the highest passage score
    # Also sort the passage list within each study by score
    scored_studies: List[Tuple[int, float, List[Passage]]] = []
    for study_id, plist in grouped.items():
        best_score = max(score for _p, score in plist)
        sorted_passages = [p for p, _score in sorted(plist, key=lambda x: -x[1])]
        scored_studies.append((study_id, best_score, sorted_passages))

    # Sort the outer list of studies by best_score
    scored_studies.sort(key=lambda x: -x[1])

    # Return top max_studies
    picked = []
    for study_id, _score, passages in scored_studies[:max_studies]:
        picked.append((study_id, passages))

    return picked


def _make_citation_line(study: Study) -> str:
    journal = study.journal or ""
    doi = f" DOI: {study.doi}" if study.doi else ""
    return f"{study.authors} ({study.year}). {study.title}. {journal}.{doi}"


def _compose_body(
    query: str,
    chosen: List[Tuple[int, List[Passage]]],
    citation_numbers: Dict[int, int],
) -> str:
    """
    Simple first-pass answer
    We stitch together short, high-level sentences from top passages
    """
    lines: List[str] = []

    # For each chosen study, add 1-2 key passages
    for study_id, passages in chosen:
        cite_num = citation_numbers[study_id]
        # Take the top passage text and trim
        main_text = passages[0].text.strip()
        if len(main_text) > 350:
            main_text = main_text[:347] + "..."

        lines.append(f"{main_text} [{cite_num}]")

    return "\n".join(lines)


def answer_query(
    query: str,
    index: TfIdfIndex,
    studies: List[Study],
    top_k_passages: int = 10,
    max_studies: int = 3,
) -> Answer:
    """
    Main entrypoint:
    - Use TF-IDF index to retrieve top passages
    - Group passages by study
    - Assign citation numbers
    - Compose answer
    """

    # Build study lookup with study_id -> study
    study_lookup = {s.id: s for s in studies}

    results = index.search(query, top_k=top_k_passages)
    if not results:
        answer_text = (
            "I couldn't find any studies in the current database that directly match your question. "
            "Try rephrasing the query or widening the topic."
        )
        return Answer(
            query=query,
            answer_text=answer_text,
            references=[],
        )

    chosen = _pick_studies_from_results(results, max_studies=max_studies)

    # Assign citation numbers for study_id
    citation_numbers: Dict[int, int] = {}
    current = 1
    for study_id, _passages in chosen:
        citation_numbers[study_id] = current
        current += 1

    # Compose return body
    body = _compose_body(query, chosen, citation_numbers)

    # Build up references list with citation index, study_id, and citation line
    references: List[Dict[str, Any]] = []
    for study_id, cite_num in citation_numbers.items():
        study = study_lookup[study_id]
        references.append(
            {
                "index": cite_num,
                "study_id": study_id,
                "citation": _make_citation_line(study),
            }
        )

    references.sort(key=lambda r: r["index"])

    return Answer(
        query=query,
        answer_text=body,
        references=references,
    )
