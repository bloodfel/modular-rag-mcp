# Rerank Benchmark Report (2026-09-22 00:13)

- Datasets: scifact, nfcorpus | Retrievers: bm25, dense, hybrid | Rerankers: none, bge, jev, llm
- Sampling: 50 BEIR **test split** queries per dataset, seed=42, retrieval depth=10, rerank depth=10
- Generated: 2026-09-22 00:13

## Quality: nDCG@10 (x100, higher is better)

| dataset/retriever | none | bge | jev | llm |
|---|---|---|---|---|
| nfcorpus/bm25 | 35.98 | 36.07 | 37.14 | 35.92 |
| nfcorpus/dense | 36.94 | 38.07 | 39.69 | 38.00 |
| nfcorpus/hybrid | 38.22 | 37.55 | 38.23 | 38.51 |
| scifact/bm25 | 64.99 | 65.27 | 73.29 | 66.16 |
| scifact/dense | 67.90 | 69.47 | 70.61 | 68.16 |
| scifact/hybrid | 67.40 | 67.19 | 72.02 | 69.68 |

## MRR@10 / Hit@10 (x100)

| dataset/retriever | none MRR | bge MRR | jev MRR | llm MRR | none Hit | bge Hit | jev Hit | llm Hit |
|---|---|---|---|---|---|---|---|---|
| nfcorpus/bm25 | 52.78 | 53.96 | 57.67 | 51.58 | 62.00 | 62.00 | 62.00 | 60.00 |
| nfcorpus/dense | 54.19 | 58.07 | 57.03 | 53.18 | 72.00 | 72.00 | 70.00 | 72.00 |
| nfcorpus/hybrid | 55.36 | 56.35 | 56.17 | 56.43 | 68.00 | 68.00 | 66.00 | 68.00 |
| scifact/bm25 | 61.93 | 62.68 | 74.45 | 64.96 | 80.00 | 80.00 | 80.00 | 80.00 |
| scifact/dense | 65.09 | 67.95 | 68.96 | 65.20 | 78.00 | 78.00 | 78.00 | 78.00 |
| scifact/hybrid | 65.35 | 64.15 | 70.89 | 67.59 | 80.00 | 80.00 | 80.00 | 80.00 |

## Speed: rerank latency (ms, p50 / p95)

| dataset/retriever | none | bge | jev | llm |
|---|---|---|---|---|
| nfcorpus/bm25 | 0 / 0 | 417 / 491 | 1801 / 5051 | 41724 / 51227 |
| nfcorpus/dense | 0 / 0 | 422 / 456 | 1885 / 4866 | 38675 / 45728 |
| nfcorpus/hybrid | 0 / 0 | 420 / 440 | 2093 / 4562 | 36646 / 43698 |
| scifact/bm25 | 0 / 0 | 595 / 617 | 1449 / 1982 | 42578 / 54997 |
| scifact/dense | 0 / 0 | 567 / 598 | 1410 / 2098 | 44963 / 53942 |
| scifact/hybrid | 0 / 0 | 566 / 630 | 1390 / 1873 | 42780 / 54966 |

## Cost: estimated rerank cost ($ total / $ per 1k queries)

Jev usage comes from `_jev_meta.tokens_in` at $0.042/MTok (output free); calls where every candidate is filtered out by min_score contribute 0. bge runs locally ($0); llm token usage is not exposed.

| dataset/retriever | none | bge | jev | llm |
|---|---|---|---|---|
| nfcorpus/bm25 | $0.0000 / $0.00 | $0.0000 / $0.00 | $0.0066 / $0.13 | $0.0000 / $0.00 |
| nfcorpus/dense | $0.0000 / $0.00 | $0.0000 / $0.00 | $0.0083 / $0.17 | $0.0000 / $0.00 |
| nfcorpus/hybrid | $0.0000 / $0.00 | $0.0000 / $0.00 | $0.0084 / $0.17 | $0.0000 / $0.00 |
| scifact/bm25 | $0.0000 / $0.00 | $0.0000 / $0.00 | $0.0088 / $0.18 | $0.0000 / $0.00 |
| scifact/dense | $0.0000 / $0.00 | $0.0000 / $0.00 | $0.0085 / $0.17 | $0.0000 / $0.00 |
| scifact/hybrid | $0.0000 / $0.00 | $0.0000 / $0.00 | $0.0086 / $0.17 | $0.0000 / $0.00 |

## Reliability: rerank failures (fallback to original order)

| dataset/retriever | none | bge | jev | llm |
|---|---|---|---|---|
| nfcorpus/bm25 | 0 | 0 | 0 | 0 |
| nfcorpus/dense | 0 | 0 | 0 | 0 |
| nfcorpus/hybrid | 0 | 0 | 0 | 0 |
| scifact/bm25 | 0 | 0 | 0 | 0 |
| scifact/dense | 0 | 0 | 0 | 0 |
| scifact/hybrid | 0 | 0 | 1 | 0 |

## Notes & caveats

- [WARN] Small sample: 50 queries per dataset - differences under ~2 nDCG points are within noise at this size.
- Caveat: BEIR corpus texts were truncated to 5000 chars (title+body) at ingest; this affects <0.25% of documents and may slightly depress recall on the longest documents.
- ir_datasets beir/scifact/test
- scifact/hybrid x jev: 1 rerank failures fell back to the original order
- scifact: rerank wall time 1320.8s (6 workers, 50 queries)
- ir_datasets beir/nfcorpus/test
- nfcorpus: rerank wall time 1102.5s (6 workers, 50 queries)
