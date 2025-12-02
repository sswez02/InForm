from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pdf_ingest import pdf_to_study_json


def slugify(text: str) -> str:
    return (
        "".join(c.lower() if c.isalnum() else "-" for c in text)
        .strip("-")
        .replace("--", "-")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a study PDF into JSON format.")
    # Optional arguments ( pdf, id, title, authors, year, doi, journal, rating, tags, training level, output directory)
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file.")
    parser.add_argument("--id", type=int, required=True, help="Numeric study ID.")
    parser.add_argument("--title", type=str, required=True, help="Study title.")
    parser.add_argument("--authors", type=str, required=True, help="Authors string.")
    parser.add_argument("--year", type=int, required=True, help="Publication year.")
    parser.add_argument("--doi", type=str, default=None, help="DOI (optional).")
    parser.add_argument("--journal", type=str, default=None, help="Journal name.")
    parser.add_argument(
        "--rating",
        type=float,
        default=4.0,
        help="Quality rating (0–5, subjective curation).",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Comma-separated tags, e.g. 'creatine,safety,long-term'",
    )
    parser.add_argument(
        "--training-status",
        type=str,
        default="unknown",
        choices=["untrained", "trained", "mixed", "athletes", "unknown"],
        help="Participant training status.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/studies",
        help="Directory to write JSON file into.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    study_dict = pdf_to_study_json(
        pdf_path=pdf_path,
        study_id=args.id,
        title=args.title,
        authors=args.authors,
        year=args.year,
        doi=args.doi,
        journal=args.journal,
        rating=args.rating,
        tags=tags,
        training_status=args.training_status,
    )

    # File name: 001_title-slug.json
    fname = f"{args.id:03d}_{slugify(args.title) or 'study'}.json"
    out_path = out_dir / fname

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(study_dict, f, ensure_ascii=False, indent=2)

    print(f"Wrote study JSON to {out_path}")


if __name__ == "__main__":
    main()
