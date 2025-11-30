from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Study:
    id: int
    title: str
    authors: str
    year: int
    doi: Optional[str]
    journal: Optional[str]
    rating: float
    tags: List[str] = field(default_factory=list)


@dataclass
class Passage:
    id: int
    study_id: int
    section: str
    text: str
