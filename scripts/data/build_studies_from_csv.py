import csv
import json
from pathlib import Path
from typing import List

from src.ft.pdf_ingest import pdf_to_study_json


PDF_ROOT = Path("data/pdfs")
CSV_PATH = Path("data/metadata/studies_master.csv")
OUT_DIR = Path("data/studies")


def parse_tags(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(";") if t.strip()]


def main() -> None:
    if not PDF_ROOT.exists():
        raise FileNotFoundError(f"PDF folder not found: {PDF_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")

        for row in reader:
            bucket = row.get("bucket", "").strip() or None
            study_id = int(row["study_id"])
            pdf_filename = row["pdf_filename"].strip()
            title = (row.get("title") or "").strip() or None
            authors = (row.get("authors") or "").strip() or None
            year_raw = (row.get("year") or "").strip()
            doi = (row.get("doi") or "").strip() or None
            training_status = (row.get("training_status") or "").strip() or "unknown"
            main_tags_raw = (row.get("main_tags") or "").strip()
            notes = (row.get("notes") or "").strip() or None
            outcome_types_text = (row.get("outcome_types") or "").strip()

            year = int(year_raw) if year_raw.isdigit() else None
            main_tags = parse_tags(main_tags_raw)

            pdf_path = PDF_ROOT / pdf_filename
            if not pdf_path.exists():
                raise FileNotFoundError(f"Missing PDF for study {study_id}: {pdf_path}")

            study = pdf_to_study_json(
                pdf_path=pdf_path,
                study_id=study_id,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                journal=None,
                rating=4.0,
                tags=main_tags if main_tags else None,
                training_status=training_status,
            )

            if bucket:
                study["bucket"] = bucket  # A/B/C/D

            if notes:
                study["notes"] = notes

            if outcome_types_text:
                study.setdefault("outcomes", {})
                study["outcomes"]["primary_human"] = outcome_types_text

            out_path = OUT_DIR / f"{study_id:03}.json"
            with out_path.open("w", encoding="utf-8") as out_f:
                json.dump(study, out_f, ensure_ascii=False, indent=2)

            print(f"Wrote {out_path}")

    print("All studies converted")


if __name__ == "__main__":
    main()
