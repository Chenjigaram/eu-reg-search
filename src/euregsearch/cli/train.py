from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..provenance import ArticleRef
from ..qa.judgements import Judgement, evaluation_articles, training_is_disjoint
from ..train.finetune import TrainConfig, train
from ..train.pairs import build_cross_lingual_pairs, build_synthetic_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the domain embedder")
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/e5-small-reg"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--judgements", type=Path, default=Path("data/processed/judgements.jsonl"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0, help="first training pair to use")
    parser.add_argument("--limit", type=int, default=None, help="how many pairs this chunk trains on")
    parser.add_argument("--init-from", type=Path, default=None, help="resume from a saved model")
    parser.add_argument("--max-seq-length", type=int, default=192)
    args = parser.parse_args()

    lines = args.articles.read_text(encoding="utf-8").splitlines()
    refs = [ArticleRef.model_validate_json(line) for line in lines if line.strip()]

    judgement_lines = args.judgements.read_text(encoding="utf-8").splitlines()
    judgements = [Judgement.model_validate_json(line) for line in judgement_lines if line.strip()]
    held = evaluation_articles(judgements)

    synthetic = build_synthetic_pairs(refs, exclude=held)
    if not training_is_disjoint(judgements, [key for _a, _p, key in synthetic]):
        raise SystemExit("ABORT: synthetic training questions overlap evaluation articles.")
    pairs = build_cross_lingual_pairs(refs) + synthetic
    total = len(pairs)
    chunk = pairs[args.offset : args.offset + args.limit] if args.limit else pairs[args.offset :]
    print(f"pairs {args.offset}..{args.offset + len(chunk)} of {total} "
          f"({len(synthetic)} synthetic, {len(held)} evaluation articles excluded)")
    pairs = chunk
    config = TrainConfig(output_dir=args.out, epochs=args.epochs, threads=args.threads,
                         max_seq_length=args.max_seq_length, init_from=args.init_from)
    print(json.dumps(train(config, pairs), indent=2))


if __name__ == "__main__":
    main()
