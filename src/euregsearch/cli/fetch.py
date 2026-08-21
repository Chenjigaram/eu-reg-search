from __future__ import annotations

import argparse
from pathlib import Path

from ..corpus.instruments import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EUR-Lex instruments and segment them into articles")
    parser.add_argument("--cache", type=Path, default=Path("data/raw/eurlex"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--retrieved", default="2026-08-21")
    args = parser.parse_args()
    print(f"{build_corpus(args.cache, args.out, args.retrieved)} articles written to {args.out}")


if __name__ == "__main__":
    main()
