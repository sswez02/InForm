from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Any

from .models import Study, Passage
from .indexer import TfIdfIndex
from .text_utils import tokenize
from .retriever import Retriever


Mode = Literal["beginner", "intermediate"]


@dataclass
class Answer:
    mode: Mode
    query: str
    answer_text: str
    references: List[Dict[str, Any]]  # {index, citation, study_id}
    confidence: str  # "high", "medium", "low"


def _pick_studies_from_results(
    results: List[Tuple[Passage, float]],
    study_lookup: Dict[int, Study],
    mode: Mode,
    query_tokens: List[str],
    max_studies: int = 3,
) -> Tuple[List[Tuple[int, List[Passage]]], List[float]]:
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

        study = study_lookup[study_id]

        training_status = getattr(study, "training_status", "mixed")
        # Weights for scoring
        wt_mode = mode_training_weight(
            mode, training_status
        )  # beginner/intermediate studies
        wt_rate = rating_weight(study.rating)  # rating
        wt_recent = recency_weight(study.year)  # publication year
        wt_tags = tag_weight(query_tokens, study.tags)  # study tags <> query match
        wt_outcome = outcome_weight(
            query=" ".join(query_tokens), outcomes=getattr(study, "outcomes", None)
        )

        final_score = best_score * wt_mode * wt_rate * wt_recent * wt_tags * wt_outcome

        scored_studies.append((study_id, final_score, sorted_passages))

    # Sort the outer list of studies by final score
    scored_studies.sort(key=lambda x: -x[1])

    # Take top max_studies and split into (chosen, scores)
    picked: List[Tuple[int, List[Passage]]] = []
    all_scores: List[float] = []

    for study_id, score, passages in scored_studies[:max_studies]:
        picked.append((study_id, passages))
        all_scores.append(score)

    return picked, all_scores


def _make_citation_line(study: Study) -> str:
    journal = study.journal or ""
    doi = f" DOI: {study.doi}" if study.doi else ""
    return f"{study.authors} ({study.year}). {study.title}. {journal}.{doi}"


def mode_training_weight(mode: Mode, training_status: str) -> float:
    if mode == "beginner":
        if training_status == "untrained":
            return 1.5
        if training_status == "mixed":
            return 1.1
        return 1.0
    else:
        if training_status in {"trained", "athletes"}:
            return 1.5
        if training_status == "mixed":
            return 1.1
        return 1.0


def recency_weight(year: int) -> float:
    CURRENT_YEAR = 2026
    age = CURRENT_YEAR - year

    if age <= 2:
        return 1.3
    elif age <= 5:
        return 1.15
    elif age <= 10:
        return 1.05
    else:
        return 1.0


def rating_weight(rating: float) -> float:
    # 0–5 rating => 1.0–1.4
    return 1.0 + (rating / 5.0) * 0.4


TAG_SYNONYMS: Dict[str, List[str]] = {
    "creatine": ["creatine", "monohydrate", "cr"],
    "hypertrophy": ["hypertrophy", "muscle growth"],
    "strength": ["strength", "power"],
    "frequency": ["frequency", "volume"],
    "hiit": ["hiit", "interval"],
    "vo2": ["vo2", "vo2max", "oxygen"],
}

OUTCOME_KEYWORDS: Dict[str, List[str]] = {
    "strength": ["strength", "1rm", "one rep max", "power"],
    "hypertrophy": ["hypertrophy", "muscle growth", "muscle size"],
    "vo2": ["vo2", "vo2max", "aerobic", "cardio", "oxygen uptake"],
    "body-composition": ["body composition", "fat mass", "lean mass", "body fat"],
}


def outcome_weight(query: str, outcomes: Dict[str, Any] | None) -> float:
    """
    Boost studies whose primary and secondary outcomes match what the user is asking about
    """

    if not outcomes:
        return 1.0

    tokens = " ".join(tokenize(query)).lower()

    primary = set(outcomes.get("primary", []))
    secondary = set(outcomes.get("secondary", []))

    boost = 1.0

    for outcome, words in OUTCOME_KEYWORDS.items():
        if any(word in tokens for word in words):
            if outcome in primary:
                boost += 0.3
            elif outcome in secondary:
                boost += 0.1

    return min(boost, 1.5)


def tag_weight(query_tokens: List[str], study_tags: List[str]) -> float:
    score = 1.0
    matched = 0

    qset = set(query_tokens)

    # Loop through study tags
    for tag in study_tags:
        tag_synonyms = TAG_SYNONYMS.get(tag, [tag])
        # If the query contains any tag synonyms derived from study tags
        if any(tok in qset for tok in tag_synonyms):
            matched += 1

    if matched == 0:
        return 1.0

    # Each match adds +10%, but cap at +30%
    return min(1.0 + matched * 0.1, 1.3)


def simplify_text_for_beginner(text: str) -> str:
    # Beginner version: remove jargon, shorten sentences
    replacements = {
        "hypertrophy": "muscle growth",
        "placebo-controlled": "compared to a non-active group",
        "resistance-trained individuals": "people who already lift weights",
        "untrained participants": "beginners",
        "significant increases": "noticeable improvements",
        "strength performance": "strength",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def _compose_body(
    query: str,
    chosen: List[Tuple[int, List[Passage]]],
    citation_numbers: Dict[int, int],
    mode: Mode,
) -> str:
    """
    Simple first-pass answer
    We stitch together short, high-level sentences from top passages
    """
    lines: List[str] = []

    if mode == "beginner":
        lines.append("Beginner mode: I'll keep the explanation simple.")
    else:
        lines.append("Intermediate mode: Using more technical language.")

    lines.append("")

    # For each chosen study, add 1-2 key passages
    for study_id, passages in chosen:
        cite_num = citation_numbers[study_id]
        # Take the top passage text and trim
        main_text = passages[0].text.strip()
        if len(main_text) > 350:
            main_text = main_text[:347] + "..."

            # If in beginner mode, simplify wording
        if mode == "beginner":
            main_text = simplify_text_for_beginner(main_text)

        lines.append("")
        lines.append(f"{main_text} [{cite_num}]")

    return "\n".join(lines)


def answer_query(
    mode: Mode,
    query: str,
    retriever: Retriever,
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

    query_tokens = tokenize(query)

    results = retriever.search(query, top_k=top_k_passages)

    if not results:
        answer_text = (
            "I couldn't find any studies in the current database that directly match your question. "
            "Try rephrasing the query or widening the topic."
        )
        return Answer(
            mode=mode,
            query=query,
            answer_text=answer_text,
            references=[],
            confidence="low",
        )

    chosen, all_scores = _pick_studies_from_results(
        results,
        study_lookup=study_lookup,
        mode=mode,
        query_tokens=query_tokens,
        max_studies=max_studies,
    )

    # No study score
    if not all_scores:
        confidence = "low"
    # One study score
    elif len(all_scores) == 1:
        top = all_scores[0]
        second = 0.0
        ratio = float("inf")
        confidence = "high"

        print(
            f"Top score: {top}, second: {second}, ratio: {ratio}, confidence: {confidence}"
        )
    # Mutliple study scores
    else:
        top = all_scores[0]
        second = all_scores[1]
        ratio = top / (second + 1e-6)

        # Compare top to second
        if ratio > 1.3:
            confidence = "high"
        elif ratio > 1.1:
            confidence = "medium"
        else:
            confidence = "low"

        print(
            f"Top score: {top}, second: {second}, ratio: {ratio}, confidence: {confidence}"
        )

    print("DEBUG study selection:")
    for study_id, _passages in chosen:
        s = study_lookup[study_id]
        print(f"  Study {study_id}: {s.title} (training_status={s.training_status})")

    # Assign citation numbers for study_id
    citation_numbers: Dict[int, int] = {}
    current = 1
    for study_id, _passages in chosen:
        citation_numbers[study_id] = current
        current += 1

    # Compose return body
    body = _compose_body(query, chosen, citation_numbers, mode=mode)

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
        mode=mode,
        query=query,
        answer_text=body,
        references=references,
        confidence=confidence,
    )
