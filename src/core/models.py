from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    training_status: str = "mixed"
    population: Dict[str, Any] = field(default_factory=dict)
    outcomes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Passage:
    id: int
    study_id: int
    section: str
    text: str
