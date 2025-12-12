from __future__ import annotations

from typing import Protocol, List, Tuple

from src.core.models import Passage


class Retriever(Protocol):
    """
    Retriever interface

    Anything that can search over passages should implement this:
    - TF-IDF index
    - Dense embedding retriever
    - Hybrid retriever

    """

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Passage, float]]: ...
