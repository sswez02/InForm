import json
from pathlib import Path
from typing import List, Tuple

from .models import Study, Passage


def load_studies_from_dir(studies_dir: Path) -> Tuple[List[Study], List[Passage]]:
    studies: List[Study] = []
    passages: List[Passage] = []
    passage_id = 1

    for path in sorted(studies_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        study = Study(
            id=int(data["id"]),
            title=data["title"],
            authors=data["authors"],
            year=int(data["year"]),
            doi=data.get("doi"),
            journal=data.get("journal"),
            rating=float(data.get("rating", 0.0)),
            tags=data.get("tags", []),
        )

        studies.append(study)

        sections = data.get("sections", {})
        for section_name, text in sections.items():
            if not text:
                continue
            passages.append(
                Passage(
                    id=passage_id,
                    study_id=study.id,
                    section=section_name,
                    text=text,
                )
            )
            passage_id += 1

    return studies, passages
