# server/main.py
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, List, Dict, Any

from pathlib import Path

from src.core.store import StudyStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import Retriever
from src.ft.answerer import answer_query, Mode
from src.ft.generation import DomainLLM
from src.core.models import Passage


app = FastAPI(title="Evidence-Based Fitness Agent")

# Load models on startup

store = StudyStore.from_dir(Path("data/studies"))
studies = store.get_all_studies()
passages: List[Passage] = store.get_all_passages()

retriever: Retriever = HybridRetriever(tfidf_weight=0.4, dense_weight=0.6)
retriever.add_passages(passages)

# DomainLLM will fall back to tiny GPT-2 on CPU, or use LoRA model on GPU
llm = DomainLLM(
    base_model_name="meta-llama/Llama-3.1-8B-Instruct",
    lora_dir="model_lora",
    max_new_tokens=64,
    max_input_tokens=512,
)


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


class AskResponse(BaseModel):
    answer: str
    mode: str
    query: str
    backend: Literal["baseline", "llm"]
    citations: List[CitationRef]
    studies: List[Dict[str, Any]]


def build_study_dict(study) -> Dict[str, Any]:
    return {
        "id": study.id,
        "title": getattr(study, "title", None),
        "source": getattr(study, "source", None),
        "metadata": getattr(study, "metadata", {}),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    mode: Mode = req.mode
    if not req.use_llm:
        # Use existing baseline pipeline
        result = answer_query(
            mode=mode,
            query=req.query,
            retriever=retriever,
            studies=studies,
            top_k_passages=req.top_k_passages,
            max_studies=req.max_studies,
        )

        # Build response
        citation_objs: List[CitationRef] = []
        for ref in result.references:
            sid = ref.study_id
            s = next((s for s in studies if s.id == sid), None)
            citation_objs.append(
                CitationRef(
                    index=ref.index,
                    study_id=sid,
                    title=getattr(s, "title", None) if s else None,
                )
            )

        return AskResponse(
            answer=result.answer_text,
            mode=req.mode,
            query=req.query,
            backend="baseline",
            citations=citation_objs,
            studies=[
                build_study_dict(s)
                for s in studies
                if s and any(r.study_id == s.id for r in result.references)
            ],
        )

    # LLM path - use same retriever + context
    retrieval_results = retriever.search(req.query, top_k=req.top_k_passages)
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

    instruction = (
        "Answer the user's question using the provided study excerpts. "
        "Explain clearly and include inline numeric citations like [1], [2] "
        "that correspond to the studies in the context."
    )

    answer_text = llm.generate_answer(
        instruction=instruction,
        query=req.query,
        context_passages=ctx,
    )

    # Build citations list from context (indexes 1..k)
    citation_objs: List[CitationRef] = []
    for c in ctx:
        sid = c["study_id"]
        idx = c["citation_index"]
        s = next((s for s in studies if s.id == sid), None)
        citation_objs.append(
            CitationRef(
                index=idx,
                study_id=sid,
                title=getattr(s, "title", None) if s else None,
            )
        )

    return AskResponse(
        answer=answer_text,
        mode=req.mode,
        query=req.query,
        backend="llm",
        citations=citation_objs,
        studies=[
            build_study_dict(s)
            for s in studies
            if s and any(c["study_id"] == s.id for c in ctx)
        ],
    )
