from __future__ import annotations

from typing import List, Tuple, Dict, Optional

from src.core.models import Passage
from .retriever import Retriever
from .indexer import TfIdfIndex

try:
    from .dense_retriever import DenseRetriever  # type: ignore
except Exception:
    DenseRetriever = None  # type: ignore


class HybridRetriever(Retriever):
    """
    Hybrid retriever that combines sparse (TF-IDF) and dense (embeddings) scores.
    If dense retriever is unavailable, it falls back to TF-IDF only.
    """

    def __init__(
        self,
        tfidf_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.tfidf_weight = tfidf_weight
        self.dense_weight = dense_weight

        self.tfidf = TfIdfIndex()
        self.dense = DenseRetriever() if DenseRetriever is not None else None
        self.passages: List[Passage] = []

    def add_passages(self, passages: List[Passage]) -> None:
        self.passages = passages
        self.tfidf.add_passages(passages)
        self.tfidf.build()

        if self.dense is not None:
            self.dense.add_passages(passages)

    def _normalise_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}

        vals = list(scores.values())
        s_min = min(vals)
        s_max = max(vals)

        if s_max == s_min:
            return {k: 0.5 for k in scores.keys()}

        return {k: (v - s_min) / (s_max - s_min) for k, v in scores.items()}

    def _effective_weights(self) -> tuple[float, float]:
        """
        If dense isn't enabled, shift all weight to TF-IDF.
        Otherwise keep original weights (renormalised if they don't sum to 1).
        """
        dense_enabled = bool(self.dense) and bool(getattr(self.dense, "enabled", False))
        if not dense_enabled:
            return 1.0, 0.0

        total = self.tfidf_weight + self.dense_weight
        if total <= 0:
            return 0.5, 0.5
        return self.tfidf_weight / total, self.dense_weight / total

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Passage, float]]:
        if not self.passages:
            return []

        k_each = min(top_k * 2, len(self.passages))

        sparse_results = self.tfidf.search(query, top_k=k_each)

        if self.dense is not None:
            dense_results = self.dense.search(
                query, top_k=k_each
            )  # returns [] if disabled
        else:
            dense_results = []

        sparse_scores: Dict[int, float] = {p.id: s for (p, s) in sparse_results}
        dense_scores: Dict[int, float] = {p.id: s for (p, s) in dense_results}

        sparse_norm = self._normalise_scores(sparse_scores)
        dense_norm = self._normalise_scores(dense_scores)

        w_sp, w_de = self._effective_weights()

        fused_scores: Dict[int, float] = {}
        for p in self.passages:
            pid = p.id
            if pid not in sparse_norm and pid not in dense_norm:
                continue

            s_sp = sparse_norm.get(pid, 0.0)
            s_de = dense_norm.get(pid, 0.0)
            fused = w_sp * s_sp + w_de * s_de

            if fused > 0.0:
                fused_scores[pid] = fused

        if not fused_scores:
            return []

        sorted_ids = sorted(fused_scores.keys(), key=lambda pid: -fused_scores[pid])
        top_ids = sorted_ids[:top_k]

        by_id = {p.id: p for p in self.passages}

        return [(by_id[pid], fused_scores[pid]) for pid in top_ids]
