# Rerank Benchmark Report (2026-09-22 16:47)

- Datasets: scifact, nfcorpus | Retrievers: hybrid | Rerankers: jev
- Sampling: 50 BEIR **test split** queries per dataset, seed=42, retrieval depth=10, rerank depth=10
- Generated: 2026-09-22 16:47

## Quality: nDCG@10 (x100, higher is better)

| dataset/retriever | jev |
|---|---|
| nfcorpus/hybrid | 31.49 |
| scifact/hybrid | 61.51 |

## MRR@10 / Hit@10 (x100)

| dataset/retriever | jev MRR | jev Hit |
|---|---|---|
| nfcorpus/hybrid | 48.67 | 54.00 |
| scifact/hybrid | 63.00 | 66.00 |

## Speed: rerank latency (ms, p50 / p95)

| dataset/retriever | jev |
|---|---|
| nfcorpus/hybrid | 1144 / 1421 |
| scifact/hybrid | 1168 / 1393 |

## Cost: estimated rerank cost ($ total / $ per 1k queries)

Jev usage comes from `_jev_meta.tokens_in` at $0.042/MTok (output free); llm usage from `_llm_meta` priced per model, USD per MTok (input/output): glm-4-flash $0.0/$0.0, deepseek-chat $0.28/$0.42, deepseek-reasoner $0.55/$2.19 — unlisted models count $0. bge runs locally ($0).

| dataset/retriever | jev |
|---|---|
| nfcorpus/hybrid | $0.0065 / $0.13 |
| scifact/hybrid | $0.0067 / $0.13 |

## Reliability: rerank failures (fallback to original order)

| dataset/retriever | jev |
|---|---|
| nfcorpus/hybrid | 0 |
| scifact/hybrid | 0 |

## Notes & caveats

- [WARN] Small sample: 50 queries per dataset - differences under ~2 nDCG points are within noise at this size.
- Caveat: BEIR corpus texts were truncated to 5000 chars (title+body) at ingest; this affects <0.25% of documents and may slightly depress recall on the longest documents.
- ir_datasets beir/scifact/test
- scifact: rerank wall time 10.5s (6 workers, 50 queries)
- ir_datasets beir/nfcorpus/test
- nfcorpus: rerank wall time 10.4s (6 workers, 50 queries)
