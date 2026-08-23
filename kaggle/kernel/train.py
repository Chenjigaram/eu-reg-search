"""Fine-tune a multilingual embedder for EU financial regulation retrieval, on a Kaggle GPU.

Mirrors src/euregsearch/train/ exactly: same pair construction, same holdouts, same loss.
Kernels cannot import the package, so the pair logic is restated here and asserted against
the same invariants the local test suite enforces.
"""
import json
import os
import random
import re
import time
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import AutoTokenizer

OUT = Path("/kaggle/working")


def find_data() -> Path:
    """Locate the corpus wherever Kaggle mounted it, and say what was found."""
    root = Path("/kaggle/input")
    print("mounted inputs:", [str(p) for p in root.glob("*")], flush=True)
    for candidate in root.rglob("articles.jsonl"):
        print(f"using corpus at {candidate.parent}", flush=True)
        return candidate.parent
    raise SystemExit(f"articles.jsonl not found under {root}; contents: {list(root.rglob('*'))[:20]}")


DATA = find_data()
HELD_OUT_INSTRUMENT = "32017R0653"
HELD_OUT_DIRECTIONS = (("nl", "en"), ("de", "fr"))
DIMENSIONS = [384, 256, 128, 64]
MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 80
MIN_TAIL_TOKENS = 40
TRAIN_SEQ_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def chunk_text(text, size=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    """Articles run far past the model window -- the median is 386 tokens and MiFID II
    Article 4 is 4955. Training on a truncated article while retrieval scores chunks means
    the model never sees the text it is later asked to match."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= size:
        return [text]
    step = size - overlap
    min_tail = max(1, min(MIN_TAIL_TOKENS, size // 4))
    pieces = []
    for start in range(0, len(ids), step):
        window = ids[start:start + size]
        if len(window) < min_tail and pieces:
            break
        pieces.append(tokenizer.decode(window))
        if start + size >= len(ids):
            break
    return pieces


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

# An unbounded cross-lingual build yields up to ten ordered directions per article and
# drowns out question-to-passage signal -- the first run was 76% alignment pairs and made
# same-language retrieval worse. Cap alignment, and use Inverse Cloze Task pairs instead of
# formulaic templates, which only teach the model to match article numbers.
CROSS_PER_ARTICLE = 2
ICT_PER_PASSAGE = 1
SENTENCE = re.compile(r"(?<=[.;:])\s+")
rng = random.Random(42)

pairs = []
for key, by_language in groups.items():
    directions = [d for d in permutations(by_language, 2) if d not in HELD_OUT_DIRECTIONS]
    if len(directions) > CROSS_PER_ARTICLE:
        directions = rng.sample(directions, CROSS_PER_ARTICLE)
    for source, target in directions:
        left = chunk_text(by_language[source]["text"])
        right = chunk_text(by_language[target]["text"])
        for i in range(min(len(left), len(right))):
            pairs.append((left[i], right[i], key))
cross = len(pairs)

for key, by_language in groups.items():
    if key in held:
        continue
    for ref in by_language.values():
        for passage in chunk_text(ref["text"]):
            sentences = [x.strip() for x in SENTENCE.split(passage) if len(x.split()) >= 8]
            if len(sentences) < 2:
                continue
            for chosen in rng.sample(sentences, min(ICT_PER_PASSAGE, len(sentences))):
                remainder = " ".join(x for x in sentences if x != chosen)
                if remainder:
                    pairs.append((chosen, remainder, key))

synthetic_keys = {k for _a, _p, k in pairs[cross:]}
assert not (synthetic_keys & held), "LEAK: synthetic questions describe evaluation articles"
assert not any(k[0] == HELD_OUT_INSTRUMENT for _a, _p, k in pairs), "LEAK: held-out instrument in training"
print(f"pairs={len(pairs)} (cross_lingual={cross}, ict={len(pairs) - cross}, "
      f"question_like={100 * (len(pairs) - cross) // len(pairs)}%); holdouts verified", flush=True)

examples = [{"anchor": f"query: {a}", "positive": f"passage: {p}"} for a, p, _k in pairs]

def usable_device() -> str:
    """torch.cuda.is_available() is not enough: Kaggle often assigns a P100 (sm_60)
    while the installed PyTorch only ships kernels for sm_70+. Probe with a real op."""
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    try:
        (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum().item()
        print(f"GPU usable: {name}", flush=True)
        return "cuda"
    except Exception as error:
        print(f"GPU {name} present but unusable ({type(error).__name__}); falling back to CPU", flush=True)
        return "cpu"


device = usable_device()
on_gpu = device == "cuda"
if not on_gpu:
    # The HF Trainer re-detects CUDA independently of the model's device, so hiding the
    # device is the only reliable way to keep an unusable GPU out of the training loop.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(f"device={device}", flush=True)

model = SentenceTransformer(MODEL_NAME, device=device)
model.max_seq_length = TRAIN_SEQ_LENGTH
loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=DIMENSIONS)

args = SentenceTransformerTrainingArguments(
    output_dir=str(OUT / "checkpoints"),
    num_train_epochs=3 if on_gpu else 2,
    per_device_train_batch_size=64 if on_gpu else 16,
    learning_rate=2e-5,
    warmup_steps=0.05,
    logging_steps=25,
    save_strategy="no",
    fp16=on_gpu,
    use_cpu=not on_gpu,
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
    "ict_pairs": len(pairs) - cross,
    "epochs": args.num_train_epochs,
    "batch_size": args.per_device_train_batch_size,
    "dimensions": DIMENSIONS,
    "max_seq_length": TRAIN_SEQ_LENGTH,
    "train_seconds": round(duration, 1),
}
(OUT / "training_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
