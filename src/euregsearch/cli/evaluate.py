from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.baselines import BM25Retriever
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
        search = BM25Retriever(refs).search
    else:
        from ..evaluation.report import dense_search_fn
        from ..index.embedder import Embedder
        from ..index.store import VectorStore

        embedder = Embedder(args.model, dimensions=args.dimensions, threads=args.threads)
        store = VectorStore()
        store.add(refs, embedder.encode([r.text for r in refs], is_query=False))
        search = dense_search_fn(embedder, store)

    name = args.name or args.system
    result = evaluate_system(name, search, judgements)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(json.dumps(result.summary(), indent=2))
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
