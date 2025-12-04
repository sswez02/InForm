from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.models import Passage
from .retriever import Retriever


class DenseRetriever(Retriever):
    """
    Dense (embedding-based) retriever over passages using a SentenceTransformer model
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.passages: List[Passage] = []
        self.embeddings: np.ndarray | None = None

    def add_passages(self, passages: List[Passage]) -> None:
        """
        Store passages and build an embedding matrix
        Call this once after loading
        """
        self.passages = passages
        texts = [p.text for p in passages]
        if not texts:
            self.embeddings = None
            return
        # (N, d) embedding matrix
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # L2 norm
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        # Embedding normalised to unit length
        self.embeddings = emb / norms

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Passage, float]]:
        """
        Embed the query and return top_k (Passage, score) pairs, where score is cosine similarity.
        """
        if not self.passages or self.embeddings is None:
            return []

        # (N, d) embedding matrix
        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        # Normalisation
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        # Cosine similarity = dot product
        scores = np.dot(self.embeddings, q_emb)

        # Top k indexes sorted
        top_k = min(top_k, len(self.passages))
        top_k_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        results: List[Tuple[Passage, float]] = []

        for idx in top_k_idx:
            results.append((self.passages[idx], float(scores[idx])))

        return results
