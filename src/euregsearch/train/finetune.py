from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DIMENSIONS = (384, 256, 128, 64)


@dataclass
class TrainConfig:
    model_name: str = "intfloat/multilingual-e5-small"
    output_dir: Path = Path("runs/e5-small-reg")
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    dimensions: tuple[int, ...] = field(default_factory=lambda: DEFAULT_DIMENSIONS)
    threads: int = 4
    max_seq_length: int = 512
    init_from: Path | None = None


def to_examples(pairs: list[tuple[str, str, tuple[str, str]]]) -> list[dict]:
    return [{"anchor": f"query: {anchor}", "positive": f"passage: {positive}"}
            for anchor, positive, _key in pairs]


def train(config: TrainConfig, pairs: list[tuple[str, str, tuple[str, str]]]) -> dict:
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    torch.set_num_threads(config.threads)
    source = str(config.init_from) if config.init_from else config.model_name
    model = SentenceTransformer(source, device="cpu")
    model.max_seq_length = config.max_seq_length
    dataset = Dataset.from_list(to_examples(pairs))
    loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model),
                          matryoshka_dims=list(config.dimensions))

    args = SentenceTransformerTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=0.05,
        logging_steps=20,
        save_strategy="no",
        use_cpu=True,
        report_to=[],
    )

    started = time.time()
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    trainer.train()
    duration = time.time() - started

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(config.output_dir / "model"))
    summary = {"model": source, "pairs": len(pairs), "epochs": config.epochs,
               "batch_size": config.batch_size, "dimensions": list(config.dimensions),
               "train_seconds": round(duration, 1)}
    (config.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
