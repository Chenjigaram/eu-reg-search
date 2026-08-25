from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..evaluation.late import late_factory
from ..evaluation.report import dense_factory
from ..evaluation.runner import evaluate_system
from ..index.embedder import Embedder
from ..provenance import ArticleRef
from ..qa.judgements import Judgement
from .evaluate import load

CONFIGURATIONS = {
    "pooled-whole": ("pooled", False),
    "pooled-chunked": ("pooled", True),
    "late-whole": ("late", False),
    "late-chunked": ("late", True),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pooling versus late interaction, with and without chunking")
    parser.add_argument("configuration", choices=sorted(CONFIGURATIONS))
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--judgements", type=Path, default=Path("data/processed/judgements.jsonl"))
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    refs = load(args.articles, ArticleRef)
    judgements = load(args.judgements, Judgement)
    aggregation, chunked = CONFIGURATIONS[args.configuration]

    embedder = Embedder(args.model, threads=args.threads)
    if aggregation == "pooled":
        factory = dense_factory(embedder, refs, chunked=chunked)
    else:
        factory = late_factory(embedder, refs, chunked=chunked)

    started = time.perf_counter()
    result = evaluate_system(args.configuration, factory, judgements)
    elapsed = time.perf_counter() - started

    summary = result.summary()
    summary["aggregation"] = aggregation
    summary["chunked"] = chunked
    summary["seconds"] = round(elapsed, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"ablation-{args.configuration}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
