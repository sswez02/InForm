from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from .models import Study, Passage


def load_studies_from_dir(studies_dir: Path) -> Tuple[List[Study], List[Passage]]:
    studies: List[Study] = []
    passages: List[Passage] = []
    passage_id = 1

    for path in sorted(studies_dir.glob("*.json")):
        # DEBUG: show which file we're reading
        print(f"Loading JSON: {path}")

        with path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON in {path}: {e}") from e

        outcomes = data.get("outcomes", {})

        study = Study(
            id=data["id"],
            title=data["title"],
            authors=data["authors"],
            year=data["year"],
            doi=data.get("doi"),
            journal=data.get("journal"),
            rating=data.get("rating", 0.0),
            tags=data.get("tags", []),
            training_status=data.get("population", {}).get(
                "training_status", "unknown"
            ),
            population=data.get("population", {}),
            outcomes=outcomes,
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
