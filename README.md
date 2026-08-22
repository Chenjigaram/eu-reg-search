# eu-reg-search

Ask a question in English, get back the exact provision of EU financial regulation that answers
it — in Dutch, German or French — with a link that proves it.

Evaluated against relevance judgements published by the regulator itself.

## What it does

> *"When does an appropriateness assessment apply rather than a suitability assessment?"*

returns MiFID II **Article 25**, in whichever language you need, with the verbatim text, the
CELEX id, the version indexed, and a EUR-Lex deep link anchored to that article.

Three properties make this worth building rather than buying:

**No generation step.** The text returned is byte-identical to EUR-Lex. Hallucination is
structurally absent rather than mitigated — the system's only claim is *this provision is
relevant*, and one click verifies it.

**Cross-lingual by construction.** A Dutch compliance officer finds an English provision and the
reverse. Lexical search cannot do this; it scores 0.095 where retrieval needs to work.

**Verifiable relevance.** Labels come from ESMA's own Q&A citations, not from generated queries
scored against themselves.

## Results

564 evaluation queries from 141 ESMA question-to-article pairs. nDCG@10.

| System | Overall | same-language | cross-lingual |
| --- | --- | --- | --- |
| **Hybrid — lexical + dense** | **0.182** | 0.282 | **0.149** |
| `multilingual-e5-small` dense | 0.154 | 0.236 | 0.127 |
| BM25 lexical | 0.152 | **0.322** | 0.095 |
| fine-tuned embedder | 0.146 | 0.171 | 0.138 |

**The hybrid beats the best single system by 19%**, and posts the best cross-lingual score of
anything tested.

That result was visible in the slices long before it was built. BM25 beats the embedder by 36% on
same-language retrieval — legal terminology is distinctive and lexical matching is strong when the
languages agree. The embedder beats BM25 by 33% cross-lingually, which BM25 cannot do at all.
Neither wins overall; fusing them by reciprocal rank does.

Anyone reporting only the overall 0.152 versus 0.154 for the two baselines would have concluded
"no difference" and been wrong twice.

## Fine-tuning did not help, and that is reported too

Contrastive fine-tuning of the embedder was tried twice and made retrieval slightly worse
(0.154 to 0.146). It improved cross-lingual retrieval (0.127 to 0.138) while degrading
same-language more (0.236 to 0.171).

The first run was 76% cross-lingual alignment pairs, teaching *"these two texts are the same
provision"* rather than *"this question is answered by this provision"*. So the mix was inverted:
alignment pairs capped, formulaic templates replaced with Inverse Cloze Task pairs drawing a real
sentence from the article as the query. Question-like pairs went from 23% to 74%, epochs doubled.

**Overall nDCG moved 0.147 to 0.146.** The obvious explanation was tested and rejected.

The remaining likely cause is corpus size: 690 trainable articles is too little for contrastive
adaptation to beat general-purpose pretraining. Domain fine-tuning is not free, and below some
corpus size it is negative — which is worth knowing before spending GPU budget on it.

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
