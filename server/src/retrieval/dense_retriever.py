from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from src.core.models import Passage
from .retriever import Retriever

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None


class DenseRetriever(Retriever):
    """
    Dense (embedding-based) retriever over passages.
    If sentence-transformers isn't installed, this retriever disables itself (returns []).
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        self.model: Optional[object] = None
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(model_name)

        self.passages: List[Passage] = []
        self.embeddings: np.ndarray | None = None

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def add_passages(self, passages: List[Passage]) -> None:
        self.passages = passages

        if not self.enabled:
            self.embeddings = None
            return

        texts = [p.text for p in passages]
        if not texts:
            self.embeddings = None
            return

        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        self.embeddings = emb / norms

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Passage, float]]:
        if (not self.enabled) or (not self.passages) or (self.embeddings is None):
            return []

        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        scores = np.dot(self.embeddings, q_emb)

        top_k = min(top_k, len(self.passages))
        top_k_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        return [(self.passages[i], float(scores[i])) for i in top_k_idx]
