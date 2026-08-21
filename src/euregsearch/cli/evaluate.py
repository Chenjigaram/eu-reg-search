from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.baselines import bm25_factory
from ..evaluation.runner import evaluate_system
from ..provenance import ArticleRef
from ..qa.judgements import Judgement


def load(path: Path, model):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a retrieval system")
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--judgements", type=Path, default=Path("data/processed/judgements.jsonl"))
    parser.add_argument("--system", choices=["bm25", "dense", "finetuned"], default="bm25")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    refs = load(args.articles, ArticleRef)
    judgements = load(args.judgements, Judgement)

    if args.system == "bm25":
        factory = bm25_factory(refs)
    else:
        from ..evaluation.report import dense_factory
        from ..index.embedder import Embedder

        embedder = Embedder(args.model, dimensions=args.dimensions, threads=args.threads)
        factory = dense_factory(embedder, refs)

    name = args.name or args.system
    result = evaluate_system(name, factory, judgements)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(json.dumps(result.summary(), indent=2))
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
