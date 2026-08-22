# EU financial regulation corpus and ESMA relevance judgements

Staging dataset for training a domain embedding model for cross-lingual retrieval over EU
financial regulation.

- `articles.jsonl` — 1,195 articles from MiFID II, MiFIR, PRIIPs and two delegated regulations,
  in English, Dutch, German and French. Each record carries CELEX id, article number, language,
  version, retrieval date and an anchor for deep linking back to EUR-Lex.
- `judgements.jsonl` — 564 evaluation queries built from 141 ESMA question-to-article pairs.
  Relevance labels come from the articles the regulator itself cites, not from generated data.
- `qa.jsonl` — the extracted Q&A entries those judgements derive from.

Source: EUR-Lex, © European Union. Reused under Commission Decision 2011/833/EU; consolidated
texts are licensed CC BY 4.0. ESMA Q&A documents are published by the European Securities and
Markets Authority for supervisory convergence.

Not legal advice.
