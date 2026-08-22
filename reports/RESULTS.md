# Retrieval results

| System | Queries | nDCG@10 | Recall@10 | MRR@10 | Provenance |
| --- | --- | --- | --- | --- | --- |
| dense-zeroshot | 564 | 0.154 | 0.238 | 0.139 | complete |
| ft-chunk1 | 564 | 0.153 | 0.213 | 0.144 | complete |
| bm25 | 564 | 0.152 | 0.235 | 0.138 | complete |
| finetuned | 564 | 0.147 | 0.224 | 0.135 | complete |

## bm25

| Slice | Queries | nDCG@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| cross_lingual | 423 | 0.095 | 0.164 | 0.081 |
| same_language | 141 | 0.322 | 0.448 | 0.309 |

## dense-zeroshot

| Slice | Queries | nDCG@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| cross_lingual | 423 | 0.127 | 0.197 | 0.114 |
| same_language | 141 | 0.236 | 0.362 | 0.215 |

## finetuned

| Slice | Queries | nDCG@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| cross_lingual | 423 | 0.141 | 0.214 | 0.128 |
| same_language | 141 | 0.168 | 0.254 | 0.155 |

## ft-chunk1

| Slice | Queries | nDCG@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| cross_lingual | 423 | 0.135 | 0.198 | 0.122 |
| same_language | 141 | 0.208 | 0.259 | 0.207 |
