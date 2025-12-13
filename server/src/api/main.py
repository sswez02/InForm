from __future__ import annotations

import os
import re

from pydantic import BaseModel
from typing import Literal, List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

from src.core.store import StudyStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ft.answerer import answer_query, Mode
from src.core.models import Passage
from .api_utils import rerank_by_recency

from src.ft.openai_llm import OpenAIDomainLLM


app = FastAPI(title="Evidence-Based Fitness Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://informapp.dev",
        "https://www.informapp.dev",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
store = StudyStore.from_dir(Path("data/studies"))
studies = store.get_all_studies()
passages: List[Passage] = store.get_all_passages()

retriever = HybridRetriever(tfidf_weight=0.4, dense_weight=0.6)
retriever.add_passages(passages)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


class AskRequest(BaseModel):
    mode: Literal["beginner", "intermediate"] = "beginner"
    query: str
    use_llm: bool = True
    top_k_passages: int = 10
    max_studies: int = 3


class CitationRef(BaseModel):
    index: int
    study_id: int
    title: str | None = None


class ConfidenceOut(BaseModel):
    value: int
    label: Literal["low", "medium", "high"]


class AskResponse(BaseModel):
    answer: str
    mode: str
    query: str
    backend: Literal["baseline", "llm"]
    citations: List[CitationRef]
    studies: List[Dict[str, Any]]
    confidence: ConfidenceOut


def build_study_dict(study) -> Dict[str, Any]:
    return {
        "id": study.id,
        "title": getattr(study, "title", None),
        "source": getattr(study, "source", None),
        "metadata": getattr(study, "metadata", {}),
    }


def extract_study_year(study) -> Optional[int]:
    year = getattr(study, "year", None)
    if isinstance(year, int):
        return year

    meta = getattr(study, "metadata", None)
    if isinstance(meta, dict):
        y = meta.get("year") or meta.get("pub_year") or meta.get("publication_year")
        if isinstance(y, int):
            return y
        if isinstance(y, str) and y.isdigit():
            return int(y)
    return None


STUDY_YEAR_BY_ID: dict[int, int] = {}
for s in studies:
    y = extract_study_year(s)
    if y is not None:
        STUDY_YEAR_BY_ID[s.id] = y

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def filter_and_renumber_citations(
    answer_text: str,
    citations: List[CitationRef],
) -> tuple[str, List[CitationRef]]:
    """
    Keep only citations that actually appear in the answer text,
    and renumber them sequentially [1], [2], ... in both the text
    and the citation list

    If the model invents a citation index that does not correspond
    to a known CitationRef (e.g. [7] when we only have 1–3),
    that citation is removed from the text
    """
    if not citations or not answer_text:
        return answer_text, citations

    valid_indices = {c.index for c in citations}

    used_order: list[int] = []
    for match in CITATION_PATTERN.finditer(answer_text):
        idx = int(match.group(1))
        if idx in valid_indices and idx not in used_order:
            used_order.append(idx)

    if not used_order:
        return answer_text, []

    index_map: dict[int, int] = {
        old: new for new, old in enumerate(used_order, start=1)
    }

    def _replace(m: re.Match) -> str:
        old_idx = int(m.group(1))
        if old_idx not in index_map:
            return ""
        new_idx = index_map[old_idx]
        return f"[{new_idx}]"

    new_answer = CITATION_PATTERN.sub(_replace, answer_text)

    new_answer = re.sub(r",\s*\]", "]", new_answer)
    new_answer = re.sub(r"\s{2,}", " ", new_answer).strip()

    new_citations: List[CitationRef] = []
    for c in citations:
        if c.index in index_map:
            new_citations.append(
                CitationRef(
                    index=index_map[c.index],
                    study_id=c.study_id,
                    title=c.title,
                )
            )

    new_citations.sort(key=lambda c: c.index)

    return new_answer, new_citations


def compute_confidence(retrieval_results: List[tuple[Any, float]]) -> Tuple[int, str]:
    """
    Turn retrieval scores into a numeric confidence + label.
    For now, we derive the numeric value from the label instead of the raw score,
    because HybridRetriever scores are not normalized to [0, 1].
    """
    if not retrieval_results:
        return 0, "low"

    scores = sorted((score for (_p, score) in retrieval_results), reverse=True)
    top = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0

    ratio = top / second if second > 0 else 1.0

    if top < 0.4 or ratio < 1.05:
        label = "low"
    elif top < 0.7 or ratio < 1.15:
        label = "medium"
    else:
        label = "high"

    if label == "high":
        value = 90
    elif label == "medium":
        value = 65
    else:
        value = 40

    return value, label


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    mode: Mode = req.mode

    raw_results = retriever.search(req.query, top_k=req.top_k_passages)
    retrieval_results = rerank_by_recency(raw_results, STUDY_YEAR_BY_ID)
    conf_value, conf_label = compute_confidence(retrieval_results)

    if not req.use_llm:
        result = answer_query(
            mode=mode,
            query=req.query,
            retriever=retriever,
            studies=studies,
            top_k_passages=req.top_k_passages,
            max_studies=req.max_studies,
        )

        citation_objs: List[CitationRef] = []
        for ref in result.references:
            if isinstance(ref, dict):
                sid = ref.get("study_id")
                idx = ref.get("index")
            else:
                sid = ref.study_id
                idx = ref.index
            if sid is None or idx is None:
                continue
            s = next((s for s in studies if s.id == sid), None)
            citation_objs.append(
                CitationRef(
                    index=int(idx),
                    study_id=int(sid),
                    title=getattr(s, "title", None) if s else None,
                )
            )

        filtered_answer, renumbered_citations = filter_and_renumber_citations(
            result.answer_text,
            citation_objs,
        )

        referenced_ids = {c.study_id for c in renumbered_citations}

        return AskResponse(
            answer=filtered_answer,
            mode=req.mode,
            query=req.query,
            backend="baseline",
            citations=renumbered_citations,
            studies=[
                build_study_dict(s) for s in studies if s and s.id in referenced_ids
            ],
            confidence=ConfidenceOut(value=conf_value, label=conf_label),
        )

    ctx = []
    for rank, (p, _score) in enumerate(retrieval_results[:3], start=1):
        ctx.append(
            {
                "study_id": p.study_id,
                "citation_index": rank,
                "section": p.section,
                "text": p.text,
            }
        )

    if req.mode == "beginner":
        style_line = (
            "Assume the user is a beginner with little resistance-training experience. "
            "Use simple, friendly language, avoid jargon, and keep the answer concise "
            "(one or two short paragraphs) with clear, practical advice."
        )
    else:
        style_line = (
            "Assume the user has some training experience (an intermediate lifter). "
            "Provide a more detailed, multi-paragraph explanation including mechanisms, "
            "key findings from the studies, and practical programming guidance. "
            "You may use standard strength-training terminology and mention approximate "
            "effect sizes or ranges when appropriate."
        )

    base_instruction = (
        "Answer the user's question using the provided study excerpts. "
        "Explain clearly and include inline numeric citations like [1], [2] "
        "that correspond to the studies in the context. "
    )
    instruction = base_instruction + style_line

    llm = OpenAIDomainLLM(
        model=OPENAI_MODEL,
        max_new_tokens=256,
    )

    try:
        answer_text = llm.generate_answer(
            instruction=instruction,
            query=req.query,
            context_passages=ctx,
        )
    except Exception as e:
        msg = str(e)
        print("LLM backend error:", repr(e), flush=True)

        if "quota" in msg.lower() or "ResourceExhausted" in msg:
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM backend is out of quota / rate-limited. "
                    "Try again later or use baseline."
                ),
            )
        raise HTTPException(
            status_code=502,
            detail=f"LLM backend error: {msg}",
        )

    citation_objs: List[CitationRef] = []
    for c in ctx:
        sid = c["study_id"]
        idx = c["citation_index"]
        s = next((s for s in studies if s.id == sid), None)
        citation_objs.append(
            CitationRef(
                index=int(idx),
                study_id=int(sid),
                title=getattr(s, "title", None) if s else None,
            )
        )

    filtered_answer, renumbered_citations = filter_and_renumber_citations(
        answer_text,
        citation_objs,
    )

    referenced_ids = {c.study_id for c in renumbered_citations}

    return AskResponse(
        answer=filtered_answer,
        mode=req.mode,
        query=req.query,
        backend="llm",
        citations=renumbered_citations,
        studies=[build_study_dict(s) for s in studies if s and s.id in referenced_ids],
        confidence=ConfidenceOut(value=conf_value, label=conf_label),
    )


@app.get("/health")
def health():
    return {"ok": True}
