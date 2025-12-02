from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import itertools
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
        section_text = heading_pattern.sub("", section_text, count=1).lstrip(": \n\t")

        sections[name] = section_text.strip()

    return sections


def split_section_into_paragraphs(section_text: str) -> List[str]:
    # Split text into paragraphs
    paras = [p.strip() for p in section_text.split("\n\n")]
    return [p for p in paras if p]


TITLE_MAX_LEN = 250


def guess_title(raw_text: str) -> str | None:
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    if not lines:
        return None

    SKIP_PREFIXES = (
        "type",
        "published",
        "doi",
        "open access",
        "openaccess",
        "edited by",
        "reviewed by",
        "keywords",
    )

    prev_upper = ""

    for line in lines:
        # Avoid very short lines
        if len(line) < 5:
            prev_upper = line.upper()
            continue

        upper = line.upper()
        lower = line.lower()

        # Skip obvious boilerplate headings
        if any(lower.startswith(prefix) for prefix in SKIP_PREFIXES):
            prev_upper = upper
            continue

        # Skip the line *after* EDITED BY / REVIEWED BY (editor names)
        if prev_upper.startswith("EDITED BY") or prev_upper.startswith("REVIEWED BY"):
            prev_upper = upper
            continue

        # Skip simple single-name lines ending with a comma
        if line.endswith(",") and " " in line and line.count(",") == 1:
            prev_upper = upper
            continue

        # Avoid overly long lines
        if len(line) > TITLE_MAX_LEN:
            prev_upper = upper
            continue

        # Avoid copyright lines
        if "©" in line or "copyright" in lower:
            prev_upper = upper
            continue

        # Avoid ALL CAPS headings
        if line.isupper() and " " in line:
            prev_upper = upper
            continue

        # Require at least 3 words to look like a proper title
        if len(line.split()) < 3:
            prev_upper = upper
            continue

        # First line that passes all checks = title
        return line

    # Fallback
    return lines[0]


def guess_authors(raw_text: str) -> str | None:
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    first_lines = list(itertools.islice(lines, 0, 100))

    for line in first_lines:
        # Require at least 2 commas (list of names)
        if line.count(",") < 2:
            continue

        # Count capitalised words
        parts = line.split()
        cap_words = sum(
            1 for part in parts if part and part[0].isalpha() and part[0].isupper()
        )

        if cap_words >= 2:
            return line
    return None


def guess_year(raw_text: str) -> int | None:
    # Only scan first ~2000 chars
    snippet = raw_text[:2000]
    candidates = re.findall(r"(19[5-9]\d|20[0-2]\d)", snippet)
    if not candidates:
        return None

    # Choose the most recent year
    years = [int(c) for c in candidates]
    return max(years) if years else None


def guess_sample_size(raw_text: str) -> int | None:
    # Only scan first ~8000 chars
    snippet = raw_text[:8000]
    #  Look for 'n = 23', 'n=45', '(n = 10)' etc.
    matches = re.findall(r"[nN]\s*=\s*(\d+)", snippet)
    if not matches:
        return None
    # pick the largest, often total sample
    nums = [int(m) for m in matches]
    return max(nums)


def guess_tags(raw_text: str) -> list[str]:
    lower = raw_text.lower()
    tags: list[str] = []

    if "creatine" in lower:
        tags.append("creatine")
    if "resistance training" in lower or "strength training" in lower:
        tags.append("resistance-training")
    if "hypertrophy" in lower or "muscle cross-sectional area" in lower:
        tags.append("hypertrophy")
    if "vo2max" in lower or "vo2 max" in lower or "oxygen uptake" in lower:
        tags.append("vo2")
    if "high-intensity interval training" in lower or "hiit" in lower:
        tags.append("hiit")

    return sorted(set(tags))


def guess_outcome_types(raw_text: str) -> list[str]:
    lower = raw_text.lower()
    out: list[str] = []

    if (
        "1rm" in lower
        or "one-repetition maximum" in lower
        or "maximal strength" in lower
    ):
        out.append("strength")
    if (
        "cross-sectional area" in lower
        or "muscle thickness" in lower
        or "hypertrophy" in lower
    ):
        out.append("hypertrophy")
    if "vo2max" in lower or "vo2 max" in lower or "oxygen uptake" in lower:
        out.append("vo2")
    if "lean body mass" in lower or "fat mass" in lower or "body composition" in lower:
        out.append("body-composition")

    return sorted(set(out))


def guess_intervention_weeks(raw_text: str) -> int | None:
    # Look for "12-week", "8 week", etc.
    matches = re.findall(r"(\d+)\s*-\s*week|\b(\d+)\s*week", raw_text.lower())
    nums = []
    for a, b in matches:
        if a:
            nums.append(int(a))
        elif b:
            nums.append(int(b))

    return max(nums) if nums else None


def pdf_to_study_json(
    pdf_path: Path,
    study_id: int,
    title: str | None = None,
    authors: str | None = None,
    year: int | None = None,
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

    # Guess metadata if not provided
    guessed_title = guess_title(raw)
    guessed_authors = guess_authors(raw)
    guessed_year = guess_year(raw)
    guessed_sample_size = guess_sample_size(raw)
    guessed_tags = guess_tags(raw)
    guessed_outcomes = guess_outcome_types(raw)
    guessed_weeks = guess_intervention_weeks(raw)

    title = title or guessed_title or "Unknown Title"
    authors = authors or guessed_authors or "Unknown Authors"
    year = year or guessed_year or 2000
    tags = tags or guessed_tags
    population = {
        "training_status": training_status,
    }
    if guessed_sample_size is not None:
        population["sample_size"] = guessed_sample_size

    sections = split_into_sections(raw)
    outcomes: Dict[str, Any] = {}
    if guessed_outcomes:
        outcomes["primary"] = guessed_outcomes
    if guessed_weeks is not None:
        outcomes["intervention_weeks"] = guessed_weeks

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
        "population": population,
        "sections": sections,
        "population": population,
        "outcomes": outcomes,
        "sections": sections,
    }
