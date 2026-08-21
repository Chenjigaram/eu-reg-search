from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..provenance import ArticleRef
from ..train.finetune import TrainConfig, train
from ..train.pairs import build_cross_lingual_pairs, build_synthetic_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the domain embedder")
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/e5-small-reg"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    lines = args.articles.read_text(encoding="utf-8").splitlines()
    refs = [ArticleRef.model_validate_json(line) for line in lines if line.strip()]
    pairs = build_cross_lingual_pairs(refs) + build_synthetic_pairs(refs)
    config = TrainConfig(output_dir=args.out, epochs=args.epochs, threads=args.threads)
    print(json.dumps(train(config, pairs), indent=2))


if __name__ == "__main__":
    main()
