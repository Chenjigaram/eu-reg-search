from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.report import comparison_table, slice_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Collate evaluation summaries")
    parser.add_argument("--dir", type=Path, default=Path("reports"))
    parser.add_argument("--out", type=Path, default=Path("reports/RESULTS.md"))
    args = parser.parse_args()

    summaries = [json.loads(p.read_text()) for p in sorted(args.dir.glob("*.json"))]
    sections = ["# Retrieval results", "", comparison_table(summaries), ""]
    for summary in summaries:
        sections += [f"## {summary['system']}", "", slice_table(summary), ""]
    report = "\n".join(sections)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
