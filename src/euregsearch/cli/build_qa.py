from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..qa.esma import MINIMUM_PAIRS, check_abort_gate, extract_entries, read_pdf_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract relevance judgements from ESMA Q&A PDFs")
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/raw/esma"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/qa.jsonl"))
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    entries = []
    for pdf in sorted(args.pdf_dir.glob("*.pdf")):
        entries += extract_entries(read_pdf_pages(pdf), args.language, pdf.stem)

    usable, passed = check_abort_gate(entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")

    print(f"usable question-to-article pairs: {usable}")
    if not passed:
        raise SystemExit(
            f"ABORT: {usable} pairs is below the {MINIMUM_PAIRS} floor fixed in the spec. "
            "The evaluation cannot support the project's claims. Stop and report this."
        )


if __name__ == "__main__":
    main()
