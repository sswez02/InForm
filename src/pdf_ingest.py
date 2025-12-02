from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re

from pypdf import PdfReader


SECTION_NAMES = [
    "abstract",
    "introduction",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
]


def extract_text_from_pdf(pdf_path: Path) -> str:

    reader = PdfReader(str(pdf_path))
    chunks: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    raw = "\n\n".join(chunks)
    # Normalise whitespace
    raw = re.sub(r"\r\n", "\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.strip()


def split_into_sections(raw_text: str) -> Dict[str, str]:

    lower = raw_text.lower()

    matches: List[Tuple[str, int]] = []
    for name in SECTION_NAMES:
        pattern = r"\b" + re.escape(name) + r"\b"
        m = re.search(pattern, lower)
        if m:
            matches.append((name, m.start()))
        if not matches:
            return {"body": raw_text}

        # Sort by position in text
        matches.sort(key=lambda x: x[1])

        # Split sections
        sections: Dict[str, str] = {}
        for i, (name, start_position) in enumerate(matches):
            end_position = matches[i + 1][1] if i + 1 < len(matches) else len(raw_text)
            section_text = raw_text[start_position:end_position].strip()

            # Strip the heading word itself from the beginning for cleanliness
            heading_pattern = re.compile(r"^" + re.escape(name), re.IGNORECASE)
            section_text = heading_pattern.sub("", section_text, count=1).lstrip(
                ": \n\t"
            )

            sections[name] = section_text.strip()

    return sections


def split_section_into_paragraphs(section_text: str) -> List[str]:
    # Split text into paragraphs
    paras = [p.strip() for p in section_text.split("\n\n")]
    return [p for p in paras if p]


def pdf_to_study_json(
    pdf_path: Path,
    study_id: int,
    title: str,
    authors: str,
    year: int,
    doi: str | None = None,
    journal: str | None = None,
    rating: float = 4.0,
    tags: List[str] | None = None,
    training_status: str = "unknown",
) -> Dict:
    """
    Given a PDF and some metadata, return the dict:

    {
      "id": ...,
      "title": ...,
      "authors": ...,
      "year": ...,
      "doi": ...,
      "journal": ...,
      "rating": ...,
      "tags": [...],
      "population": {"training_status": "..."},
      "sections": {
         "abstract": "...",
         "introduction": "...",
         ...
      }
    }
    """
    raw = extract_text_from_pdf(pdf_path)
    sections = split_into_sections(raw)

    return {
        "id": study_id,
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "journal": journal,
        "rating": rating,
        "tags": tags or [],
        "population": {"training_status": training_status},
        "sections": sections,
    }
