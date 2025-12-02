from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import Study, Passage
from .load_studies import load_studies_from_dir


@dataclass
class StudyStore:
    """
    Simple in memory store for studies and passages

    To add other implementations (DB-backed, API-backed) later
    """

    studies: List[Study]
    passages: List[Passage]
    _study_by_id: Dict[int, Study]

    @classmethod
    def from_dir(cls, studies_dir: Path) -> "StudyStore":
        studies, passages = load_studies_from_dir(studies_dir)
        study_by_id = {s.id: s for s in studies}
        return cls(studies=studies, passages=passages, _study_by_id=study_by_id)

    # Study methods
    def get_all_studies(self) -> List[Study]:
        return self.studies

    def get_study_by_id(self, study_id: int) -> Optional[Study]:
        return self._study_by_id.get(study_id)

    def iter_studies(self) -> Iterable[Study]:
        return iter(self.studies)

    # Passage methods
    def get_all_passages(self) -> List[Passage]:
        return self.passages

    def iter_passages(self) -> Iterable[Passage]:
        return iter(self.passages)

    def get_passages_for_study(self, study_id: int) -> List[Passage]:
        return [p for p in self.passages if p.study_id == study_id]
