"""Fine-tune a multilingual embedder for EU financial regulation retrieval, on a Kaggle GPU.

Mirrors src/euregsearch/train/ exactly: same pair construction, same holdouts, same loss.
Kernels cannot import the package, so the pair logic is restated here and asserted against
the same invariants the local test suite enforces.
"""
import json
import time
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

DATA = Path("/kaggle/input/eu-reg-search-corpus")
OUT = Path("/kaggle/working")
HELD_OUT_INSTRUMENT = "32017R0653"
HELD_OUT_DIRECTIONS = (("nl", "en"), ("de", "fr"))
DIMENSIONS = [384, 256, 128, 64]


def load(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


articles = load("articles.jsonl")
judgements = load("judgements.jsonl")
held = {tuple(pair) for j in judgements for pair in j["relevant"]}
print(f"articles={len(articles)} judgements={len(judgements)} evaluation_articles={len(held)}", flush=True)

groups = defaultdict(dict)
for a in articles:
    if a["celex"] == HELD_OUT_INSTRUMENT:
        continue
    groups[(a["celex"], a["article"])][a["language"]] = a

pairs = []
for key, by_language in groups.items():
    for source, target in permutations(by_language, 2):
        if (source, target) in HELD_OUT_DIRECTIONS:
            continue
        pairs.append((by_language[source]["text"], by_language[target]["text"], key))
cross = len(pairs)

for key, by_language in groups.items():
    if key in held:
        continue
    for ref in by_language.values():
        pairs.append((f"What does Article {ref['article']} of {ref['celex']} provide?", ref["text"], key))

synthetic_keys = {k for _a, _p, k in pairs[cross:]}
assert not (synthetic_keys & held), "LEAK: synthetic questions describe evaluation articles"
assert not any(k[0] == HELD_OUT_INSTRUMENT for _a, _p, k in pairs), "LEAK: held-out instrument in training"
print(f"pairs={len(pairs)} (cross_lingual={cross}, synthetic={len(pairs) - cross}); holdouts verified", flush=True)

examples = [{"anchor": f"query: {a}", "positive": f"passage: {p}"} for a, p, _k in pairs]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''}", flush=True)

model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)
model.max_seq_length = 192
loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=DIMENSIONS)

args = SentenceTransformerTrainingArguments(
    output_dir=str(OUT / "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=0.05,
    logging_steps=25,
    save_strategy="no",
    fp16=(device == "cuda"),
    report_to=[],
)

started = time.time()
SentenceTransformerTrainer(model=model, args=args, train_dataset=Dataset.from_list(examples), loss=loss).train()
duration = time.time() - started

model.save(str(OUT / "model"))
summary = {
    "base_model": "intfloat/multilingual-e5-small",
    "device": device,
    "pairs": len(pairs),
    "cross_lingual_pairs": cross,
    "synthetic_pairs": len(pairs) - cross,
    "epochs": args.num_train_epochs,
    "batch_size": args.per_device_train_batch_size,
    "dimensions": DIMENSIONS,
    "max_seq_length": 192,
    "train_seconds": round(duration, 1),
}
(OUT / "training_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
