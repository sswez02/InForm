from typing import Protocol
from src.core.models import Study
from src.retrieval.retriever import Retriever
from .answerer import Answer, Mode


class AnswerGenerator(Protocol):
    def generate(
        self,
        mode: Mode,
        query: str,
        retriever: Retriever,
        studies: list[Study],
    ) -> Answer: ...
