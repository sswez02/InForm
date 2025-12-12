from __future__ import annotations

from typing import List, Tuple, Dict

from src.core.models import Passage
from .retriever import Retriever
from .indexer import TfIdfIndex
from .dense_retriever import DenseRetriever


class HybridRetriever(Retriever):
    """
    Hybrid retriever that combines sparse (TF-IDF) and dense (embeddings) scores into a single ranking
    """

    def __init__(
        self,
        tfidf_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.tfidf_weight = tfidf_weight
        self.dense_weight = dense_weight

        self.tfidf = TfIdfIndex()
        self.dense = DenseRetriever()
        self.passages: List[Passage] = []

    def add_passages(self, passages: List[Passage]) -> None:
        self.passages = passages
        self.tfidf.add_passages(passages)
        self.tfidf.build()
        self.dense.add_passages(passages)

    def _normalise_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """
        Min-max normalise scores to [0, 1] range
        """
        if not scores:
            return {}

        vals = list(scores.values())
        s_min = min(vals)
        s_max = max(vals)

        if s_max == s_min:
            return {k: 0.5 for k in scores.keys()}

        return {k: (v - s_min) / (s_max - s_min) for k, v in scores.items()}

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Passage, float]]:
        """
        Run both TF-IDF and Dense retrieval, normalise scores, and combine them

        Returns a list of (Passage, fused_score) sorted by fused_score desc
        """
        if not self.passages:
            return []

        k_each = min(top_k * 2, len(self.passages))

        sparse_results = self.tfidf.search(query, top_k=k_each)
        dense_results = self.dense.search(query, top_k=k_each)

        # Map list of (Passage, score)” to “passage_id -> score
        sparse_scores: Dict[int, float] = {p.id: s for (p, s) in sparse_results}
        dense_scores: Dict[int, float] = {p.id: s for (p, s) in dense_results}

        # Normalise each set to [0, 1]
        sparse_norm = self._normalise_scores(sparse_scores)
        dense_norm = self._normalise_scores(dense_scores)

        # Fuse sparse and dense normalised
        fused_scores: Dict[int, float] = {}

        for p in self.passages:
            pid = p.id
            if pid not in sparse_norm and pid not in dense_norm:
                continue
            s_sp = sparse_norm.get(pid, 0.0)
            s_de = dense_norm.get(pid, 0.0)

            fused = self.tfidf_weight * s_sp + self.dense_weight * s_de
            if fused > 0.0:
                fused_scores[pid] = fused

        if not fused_scores:
            return []

        sorted_ids = sorted(fused_scores.keys(), key=lambda pid: -fused_scores[pid])

        top_ids = sorted_ids[:top_k]

        # Build Passage lookup by id
        by_id = {p.id: p for p in self.passages}

        results: List[Tuple[Passage, float]] = []
        for pid in top_ids:
            p = by_id[pid]
            results.append((p, fused_scores[pid]))

        return results
