# eu-reg-search

Ask a question in English, get back the exact provision of EU financial regulation that answers
it — in Dutch, German or French — with a link that proves it.

Evaluated against relevance judgements published by the regulator itself.

**[Results and worked examples →](https://chenjigaram.github.io/eu-reg-search/)**

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
| **Hybrid — lexical + dense** | **0.222** | **0.338** | 0.183 |
| `multilingual-e5-small` dense | 0.219 | 0.305 | **0.190** |
| BM25 lexical | 0.152 | 0.322 | 0.095 |

**Both dense systems beat BM25 by 5.5 standard errors.** The hybrid's 0.222 against dense 0.219 is
0.003 — about 0.2 standard errors on 564 queries, which is not a win and is not claimed as one.

That is itself the finding. Before chunking, fusing lexical and dense retrieval bought a real 19%
over the best single system, and it was the headline result of this repo. It no longer holds: once
the embedder could read whole articles it matched the hybrid alone, and fusion now *costs* 0.007
cross-lingually, because mixing a 0.095 system into a 0.190 one dilutes it. **The hybrid had been
compensating for a truncation bug.**

Routing by whether the query and target language match — hybrid when they agree, dense when they do
not — scores ~0.227 and is the honest design.

## The retriever was reading a fraction of each article

`multilingual-e5-small` has a 512-token window. **36 of the 63 cited articles (57%) exceed it.**
MiFID II Article 4 is 4,955 tokens — 9.7x the window, so 90% of the most-cited provision in the
corpus was invisible to the embedder. It was being scored on text it never saw.

Indexing 400-token passages with 80 tokens of overlap and scoring each article by its best passage:

| | before | after |
| --- | --- | --- |
| dense, overall | 0.154 | **0.219** (+42%) |
| dense, cross-lingual | 0.127 | **0.190** (+50%) |
| hybrid, overall | 0.182 | **0.222** (+22%) |

No new parameters, no training. The embedder had been losing same-language retrieval to BM25
because BM25 reads the whole article and it did not; chunked, it takes that slice back (0.338
against 0.322).

## Fine-tuning did not help, and the reason was not the one I gave

Contrastive fine-tuning was tried twice and made retrieval worse both times (0.154 to 0.146). The
first run was 76% cross-lingual alignment pairs, teaching *"these two texts are the same
provision"* rather than *"this question is answered by this provision"*. Inverting the mix — capping
alignment pairs, replacing formulaic templates with Inverse Cloze Task pairs — moved overall nDCG
from 0.147 to 0.146. The obvious explanation was tested and rejected.

I then attributed the failure to corpus size: 690 trainable articles being too few for contrastive
adaptation to beat general-purpose pretraining. That was a guess, and the measurement contradicts
it.

**Training ran at `max_seq_length = 192`.** 75% of articles are longer than that, and the median
article is 386 tokens — so the model was trained on roughly half of a typical provision and then
asked at inference to retrieve a 512-token representation of it. The positives it learned and the
passages it was scored against were different text.

Both ends now match: pairs are built from the same 400-token chunks the index serves, and the
training window equals the inference encoder's. The re-run is pending; whatever it returns will be
reported here, including if it fails again.

The honest state is that the corpus-size claim was never measured, and the truncation was. Domain
fine-tuning may still turn out not to pay for itself at this scale — but that has not been shown
yet, and the earlier negative result cannot carry the weight I put on it.

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
