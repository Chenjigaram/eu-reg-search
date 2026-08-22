# eu-reg-search

Cross-lingual retrieval over EU financial regulation, evaluated against relevance judgements
published by the regulator itself.

**Headline finding: fine-tuning a multilingual embedder on this corpus made retrieval worse.**
That result is reproduced twice, with two different training-data compositions, and it is the
most useful thing here.

## The task

Compliance and product teams ask questions like *"does the KID obligation apply to professional
clients?"* The answer is a specific provision in a specific instrument, and it must be citable.
MiFID II alone is 1.3 MB of dense legal prose across 24 language versions.

Retrieval is load-bearing here in a way it often is not. There is nothing to filter on — regulation
is not tabular and nobody will tag it into columns. And there is **no generation step**, so the
text returned is byte-identical to EUR-Lex: hallucination is structurally absent rather than
mitigated. The system's only claim is *this provision is relevant*, and every result links back
to the article so a reader can check.

## Results

564 evaluation queries built from 141 ESMA question-to-article pairs. nDCG@10.

| System | Overall | same-language | cross-lingual |
| --- | --- | --- | --- |
| BM25 (lexical) | 0.152 | **0.322** | 0.095 |
| `multilingual-e5-small` zero-shot | **0.154** | 0.236 | 0.127 |
| fine-tuned v1 — 76% alignment pairs | 0.147 | 0.168 | **0.141** |
| fine-tuned v2 — 74% Inverse Cloze pairs | 0.146 | 0.171 | 0.138 |

**No system wins overall, and the aggregate number hides everything.** BM25 beats the embedder by
36% on same-language retrieval; the embedder beats BM25 by 33% cross-lingually, which lexical
matching cannot do by construction. Reporting only the overall 0.152 versus 0.154 would conclude
"no difference" and be wrong twice.

### Why fine-tuning failed, and what was ruled out

The first run was 76% cross-lingual alignment pairs — teaching *"these two texts are the same
provision"* rather than *"this question is answered by this provision"*. Same-language retrieval
degraded monotonically with training: 0.236 zero-shot, 0.208 at 13% of the data, 0.168 at 100%.
Three points, one direction.

So the mix was inverted: alignment pairs capped, formulaic templates (*"What does Article 25
provide?"*, which mostly teaches article-number matching) replaced with Inverse Cloze Task pairs
that draw a real sentence from the article as the query. Question-like pairs went from 23% to 74%,
epochs doubled.

**Overall nDCG moved 0.147 to 0.146.** The obvious explanation was tested and rejected.

The remaining likely cause is corpus size: 690 trainable articles is too little for contrastive
adaptation, and training pulls the embedding space away from the general-purpose pretraining that
was doing the work. Domain fine-tuning is not free, and below some corpus size it is negative.

## What makes the evaluation trustworthy

**The relevance labels come from the regulator, not from us.** ESMA and Joint Committee Q&A
documents state a question, an answer, and the articles they rely on. That yields
`(question, language) -> [cited articles]` judgements that anyone can verify by opening the PDF.
No LLM judge, no self-generated queries scored against themselves.

Yield is itself a finding: of **546 Q&A blocks** extracted from five documents, only **141 (25%)**
cite an article resolvable to an indexed instrument. The PRIIPs consolidated Q&A yields just 4%,
because its answers cite *annexes* — the KID methodology lives there rather than in numbered
articles, and this system indexes articles only.

An abort gate refuses to proceed below 120 usable pairs. It fired at 19, survived two parser
fixes to 71, and only passed at 141 after adding three more source documents.

### Known gaps, stated rather than hidden

- **`held_out_instrument`: 0 judgements.** No PRIIPs RTS citations survived extraction.
- **`held_out_direction`: 0 judgements.** Every ESMA question is in English, so `nl→en` and
  `de→fr` never occur. The spec promised holdouts the data cannot populate.
- **64 distinct articles** are cited across 141 questions, so effective diversity is 64, not 141.
- Single seed. Only large gaps are claimed.

## Provenance

Every result carries the verbatim passage, instrument, article number, CELEX id, language,
version indexed, retrieval date, and a EUR-Lex deep link anchored to the article. Provenance
completeness is checked as a **defect, not a metric** — anything below 100% fails.

MiFID II has been amended repeatedly; consolidated text differs from the original, and citing the
wrong version is a compliance error, so the version is recorded and displayed.

## Running it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,pdf,search,train]"
python -m euregsearch.cli.fetch          # EUR-Lex via the Cellar API
python -m euregsearch.cli.build_qa       # ESMA Q&As -> judgements (aborts below 120 pairs)
python -m euregsearch.cli.evaluate --system bm25
python -m euregsearch.cli.evaluate --system dense
python -m euregsearch.cli.report
```

The EUR-Lex web UI sits behind an AWS WAF bot challenge and returns 202 with an empty body to
automated clients. `corpus/instruments.py` uses the Publications Office **Cellar** endpoint, which
is the sanctioned machine-to-machine route.

`kaggle/` contains a dataset spec and training kernel for running the fine-tune unattended.
Note that Kaggle assigns a P100 (sm_60) which the installed PyTorch cannot execute (sm_70+ only),
so the kernel probes real GPU capability with a live matmul and falls back to CPU.

## Prior work

**LEMUR** (arXiv:2602.09570) fine-tunes multilingual law embedding models for retrieval.
**MultiEURLEX** covers 65k EU laws in 23 languages for classification. This project claims no
novel technique — the contributions are the domain and the ground-truth source.

## Licence and disclaimer

MIT. Corpus text: EUR-Lex, © European Union, reused under Commission Decision 2011/833/EU;
consolidated texts CC BY 4.0. ESMA Q&A documents published for supervisory convergence.

**Not legal advice.** This retrieves provisions; a human interprets them.
