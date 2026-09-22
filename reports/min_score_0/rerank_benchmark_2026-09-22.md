# Rerank Benchmark Report (2026-09-22 16:49)

- Datasets: scifact, nfcorpus | Retrievers: hybrid | Rerankers: jev
- Sampling: 50 BEIR **test split** queries per dataset, seed=42, retrieval depth=10, rerank depth=10
- Generated: 2026-09-22 16:49

## Quality: nDCG@10 (x100, higher is better)

| dataset/retriever | jev |
|---|---|
| nfcorpus/hybrid | 38.43 |
| scifact/hybrid | 72.05 |

## MRR@10 / Hit@10 (x100)

| dataset/retriever | jev MRR | jev Hit |
|---|---|---|
| nfcorpus/hybrid | 55.37 | 68.00 |
| scifact/hybrid | 70.96 | 80.00 |

## Speed: rerank latency (ms, p50 / p95)

| dataset/retriever | jev |
|---|---|
| nfcorpus/hybrid | 1238 / 2201 |
| scifact/hybrid | 1314 / 1986 |

## Cost: estimated rerank cost ($ total / $ per 1k queries)

Jev usage comes from `_jev_meta.tokens_in` at $0.042/MTok (output free); llm usage from `_llm_meta` priced per model, USD per MTok (input/output): glm-4-flash $0.0/$0.0, deepseek-chat $0.28/$0.42, deepseek-reasoner $0.55/$2.19 — unlisted models count $0. bge runs locally ($0).

| dataset/retriever | jev |
|---|---|
| nfcorpus/hybrid | $0.0085 / $0.17 |
| scifact/hybrid | $0.0088 / $0.18 |

## Reliability: rerank failures (fallback to original order)

| dataset/retriever | jev |
|---|---|
| nfcorpus/hybrid | 0 |
| scifact/hybrid | 0 |

## Notes & caveats

- [WARN] Small sample: 50 queries per dataset - differences under ~2 nDCG points are within noise at this size.
- Caveat: BEIR corpus texts were truncated to 5000 chars (title+body) at ingest; this affects <0.25% of documents and may slightly depress recall on the longest documents.
- ir_datasets beir/scifact/test
- scifact: rerank wall time 12.3s (6 workers, 50 queries)
- ir_datasets beir/nfcorpus/test
- nfcorpus: rerank wall time 11.8s (6 workers, 50 queries)
