from typing import Protocol
from .models import Study
from .retriever import Retriever
from .answerer import Answer, Mode


class AnswerGenerator(Protocol):
    def generate(
        self,
        mode: Mode,
        query: str,
        retriever: Retriever,
        studies: list[Study],
    ) -> Answer: ...
