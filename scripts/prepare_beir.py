#!/usr/bin/env python
"""BEIR dataset preparation script for the rerank benchmark.

Loads public BEIR datasets (scifact / nfcorpus [fiqa]) via ``ir_datasets``,
ingests the corpus into a dedicated collection using the project's existing
ingestion components, and dumps queries/qrels for offline evaluation.

Artifacts per dataset ``<name>``:
    - Chroma collection ``beir-<name>``
    - BM25 index ``data/db/bm25/beir-<name>/beir-<name>_bm25.json``
    - ``data/beir/<name>/queries.json``  -> [{"query_id": str, "text": str}]
    - ``data/beir/<name>/qrels.json``    -> {query_id: {doc_id: relevance}}

``--limit`` smoke runs write to isolated ``*-smoke`` paths (collection
``beir-<name>-smoke``, ``data/beir/<name>-smoke/``,
``data/db/bm25/beir-<name>-smoke/``) so they never touch official artifacts.

Conversion rules:
    - One corpus document = one Chunk (no splitting).
    - chunk text = f"{title}\n\n{text}".strip(), truncated to
      EMBED_TEXT_MAX_CHARS (embedding-model context safety)
    - chunk metadata: source_path=f"beir://{name}/{doc_id}", chunk_index=0,
      doc_type="beir", title=<title>

Note on ir_datasets API: depending on the installed version, the base
``beir/<name>`` dataset exposes the corpus either as ``corpus_iter()`` or
``docs_iter()``, and qrels either on the base dataset or only on subsets
(``beir/<name>/{test,dev,train}``).  This script supports both layouts.

Usage:
    # Smoke test (first 200 docs)
    python scripts/prepare_beir.py --datasets scifact --limit 200

    # Full ingest (force re-ingest, upsert is idempotent)
    python scripts/prepare_beir.py --datasets scifact --force

    # Both datasets
    python scripts/prepare_beir.py --datasets scifact nfcorpus

Exit codes:
    0 - Success (all requested datasets prepared)
    1 - Partial failure (some datasets failed)
    2 - Complete failure (all datasets failed or configuration error)
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import ir_datasets

from src.core.settings import load_settings, resolve_path
from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.embedding.embedding_factory import EmbeddingFactory

EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 256
ENCODE_RETRIES = 3
ENCODE_RETRY_DELAY_S = 5.0
# nomic-embed-text has a 2048-token context; Ollama returns HTTP 500 for
# inputs exceeding it instead of truncating.  Cap at 5000 chars (~1200-1600
# tokens), in line with standard BEIR practice of truncating long docs.
EMBED_TEXT_MAX_CHARS = 5000

SUPPORTED_DATASETS = ("scifact", "nfcorpus", "fiqa")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare BEIR datasets for the rerank benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "nfcorpus"],
        choices=SUPPORTED_DATASETS,
        help="BEIR datasets to prepare (default: scifact nfcorpus)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N corpus documents (smoke testing). "
        "Artifacts go to isolated '*-smoke' paths; official full-run "
        "artifacts are never touched.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if artifacts already exist (upsert is idempotent)",
    )

    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="Path to configuration file (default: config/settings.yaml)",
    )

    return parser.parse_args()


def _corpus_iter(ds: Any) -> Iterator[Any]:
    """Yield corpus records from an ir_datasets Dataset.

    Supports both the ``corpus_iter()`` and ``docs_iter()`` API layouts and
    normalizes records to a ``doc_id/title/text`` mapping.
    """
    it: Iterable[Any] = ds.corpus_iter() if hasattr(ds, "corpus_iter") else ds.docs_iter()
    for rec in it:
        d = rec._asdict() if hasattr(rec, "_asdict") else dict(rec)
        yield {
            "doc_id": str(d["doc_id"]),
            "title": d.get("title") or "",
            "text": d.get("text") or "",
        }


def _queries(ds: Any) -> List[Dict[str, str]]:
    """Return all queries as [{"query_id", "text"}]."""
    queries = []
    for rec in ds.queries_iter():
        q = rec._asdict() if hasattr(rec, "_asdict") else dict(rec)
        queries.append({"query_id": str(q["query_id"]), "text": q["text"]})
    return queries


def _qrels(ds: Any, name: str) -> Dict[str, Dict[str, int]]:
    """Return qrels as {query_id: {doc_id: relevance}}.

    Uses the base dataset's qrels when available; otherwise unions qrels
    across the dataset's registered subsets (test/dev/train).
    """
    qrels: Dict[str, Dict[str, int]] = {}

    def _collect(iterator: Iterable[Any]) -> None:
        for rec in iterator:
            r = rec._asdict() if hasattr(rec, "_asdict") else dict(rec)
            qrels.setdefault(str(r["query_id"]), {})[str(r["doc_id"])] = int(r["relevance"])

    if hasattr(ds, "qrels_iter"):
        _collect(ds.qrels_iter())
        return qrels

    for split in ("test", "dev", "train", "val"):
        subset_id = f"beir/{name}/{split}"
        if subset_id not in ir_datasets.registry:
            continue
        _collect(ir_datasets.load(subset_id).qrels_iter())

    return qrels


def _artifact_paths(name: str, limit: Optional[int] = None) -> Dict[str, Path]:
    """Return artifact paths for a dataset.

    ``--limit`` smoke runs are isolated under ``*-smoke`` paths and a
    ``beir-<name>-smoke`` collection so they can never overwrite or poison
    the official full-run artifacts (a partial index in the official path
    would otherwise make the next full run's skip check pass silently).
    """
    suffix = "-smoke" if limit is not None else ""
    beir_dir = resolve_path(f"data/beir/{name}{suffix}")
    bm25_dir = resolve_path(f"data/db/bm25/beir-{name}{suffix}")
    return {
        "collection": f"beir-{name}{suffix}",
        "queries": beir_dir / "queries.json",
        "qrels": beir_dir / "qrels.json",
        "bm25_dir": bm25_dir,
        "bm25_index": bm25_dir / f"beir-{name}{suffix}_bm25.json",
    }


def prepare_dataset(name: str, settings: Any, limit: Optional[int], force: bool) -> bool:
    """Prepare one BEIR dataset (corpus ingest + queries/qrels dump).

    Args:
        name: BEIR dataset name (e.g. 'scifact').
        settings: Application settings.
        limit: Optional cap on ingested corpus documents.
        force: Re-ingest even if artifacts exist.

    Returns:
        True on success, False on failure.
    """
    print(f"\n{'=' * 60}")
    print(f"BEIR dataset: {name}")
    print("=" * 60)

    paths = _artifact_paths(name, limit)

    ds = None
    # Idempotency: skip only when queries.json + BM25 index exist AND the
    # index covers the full corpus.  A stale/partial index (e.g. left by an
    # interrupted run) must trigger a rebuild rather than a silent skip.
    # BM25Indexer has no public metadata accessor; _metadata is the
    # documented index structure (see bm25_indexer module docstring).
    if not force and limit is None and paths["queries"].exists() and paths["bm25_index"].exists():
        ds = ir_datasets.load(f"beir/{name}")
        expected_docs = ds.docs_count()
        indexer = BM25Indexer(index_dir=str(paths["bm25_dir"]))
        indexed_docs = (
            indexer._metadata.get("num_docs")
            if indexer.load(collection=paths["collection"])
            else None
        )
        if indexed_docs == expected_docs:
            print(f"[SKIP] Artifacts already exist for '{name}' (use --force to re-ingest)")
            print(f"       queries: {paths['queries']}")
            print(f"       bm25:    {paths['bm25_index']} ({indexed_docs} docs)")
            return True
        print(f"[STALE] BM25 index covers {indexed_docs} docs but corpus has "
              f"{expected_docs}; rebuilding '{name}' instead of skipping")

    if ds is None:
        ds = ir_datasets.load(f"beir/{name}")
    collection = paths["collection"]

    # Queries/qrels: cheap, no embedding required, always written in full.
    queries = _queries(ds)
    qrels = _qrels(ds, name)
    if not qrels:
        print(f"[FAIL] No qrels found for '{name}'")
        return False

    paths["queries"].parent.mkdir(parents=True, exist_ok=True)
    paths["queries"].write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["qrels"].write_text(
        json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] queries.json: {len(queries)} queries")
    print(f"[OK] qrels.json: {sum(len(v) for v in qrels.values())} pairs "
          f"({len(qrels)} queries)")

    # Corpus ingest.
    embedding = EmbeddingFactory.create(settings)
    dense_encoder = DenseEncoder(embedding, batch_size=EMBED_BATCH_SIZE)
    sparse_encoder = SparseEncoder()
    upserter = VectorUpserter(settings, collection_name=collection)

    total = ds.docs_count()
    if limit is not None:
        total = min(total, limit)
    num_batches = (total + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

    print(f"[INFO] Ingesting {total} docs into collection '{collection}' "
          f"(embed batch={EMBED_BATCH_SIZE}, upsert batch={UPSERT_BATCH_SIZE})")

    corpus = _corpus_iter(ds)
    if limit is not None:
        corpus = (doc for _, doc in zip(range(limit), corpus))

    term_stats: List[Dict[str, Any]] = []
    started = time.monotonic()
    ingested = 0

    batch_iter = iter(corpus)
    while True:
        window = list(itertools.islice(batch_iter, UPSERT_BATCH_SIZE))
        if not window:
            break

        chunks = [
            Chunk(
                id=f"{collection}-{doc['doc_id']}",
                text=f"{doc['title']}\n\n{doc['text']}".strip()[:EMBED_TEXT_MAX_CHARS],
                metadata={
                    "source_path": f"beir://{name}/{doc['doc_id']}",
                    "chunk_index": 0,
                    "doc_type": "beir",
                    "title": doc["title"],
                },
            )
            for doc in window
        ]

        # Retry transient embedding server errors (e.g. occasional Ollama 500s)
        # so one hiccup does not discard minutes of completed work.
        vectors = None
        for attempt in range(1, ENCODE_RETRIES + 1):
            try:
                vectors = dense_encoder.encode(chunks)
                break
            except RuntimeError as e:
                if attempt == ENCODE_RETRIES:
                    raise
                print(f"[WARN] Embedding failed (attempt {attempt}/{ENCODE_RETRIES}): {e}")
                print(f"[WARN] Retrying in {ENCODE_RETRY_DELAY_S}s ...")
                time.sleep(ENCODE_RETRY_DELAY_S)

        vector_ids = upserter.upsert(chunks, vectors)

        stats = sparse_encoder.encode(chunks)
        # Align BM25 chunk_ids with Chroma vector IDs so the SparseRetriever
        # can look up BM25 hits in the vector store after retrieval.
        for stat, vid in zip(stats, vector_ids):
            stat["chunk_id"] = vid
        term_stats.extend(stats)

        ingested += len(chunks)
        elapsed = time.monotonic() - started
        print(f"[BATCH {ingested // UPSERT_BATCH_SIZE}/{num_batches}] "
              f"{ingested}/{total} docs ingested "
              f"({elapsed:.1f}s elapsed, ~{elapsed / ingested * 1000:.0f} ms/doc)")

    if ingested == 0:
        print(f"[FAIL] No corpus documents loaded for '{name}'")
        return False

    # Build BM25 index (single build over the whole corpus).
    indexer = BM25Indexer(index_dir=str(paths["bm25_dir"]))
    indexer.build(term_stats, collection=collection)

    total_time = time.monotonic() - started
    print(f"[OK] '{name}' done: {ingested} docs, {len(queries)} queries, "
          f"{sum(len(v) for v in qrels.values())} qrel pairs in {total_time:.1f}s")
    return True


def main() -> int:
    """Main entry point. Returns process exit code."""
    args = parse_args()

    print("[*] BEIR Dataset Preparation Script")
    print("=" * 60)

    try:
        settings = load_settings(args.config)
        print(f"[OK] Configuration loaded from: {args.config}")
        print(f"[OK] Embedding provider: {settings.embedding.provider} "
              f"({settings.embedding.model})")
    except Exception as e:
        print(f"[FAIL] Failed to load configuration: {e}")
        return 2

    results = {}
    for name in args.datasets:
        try:
            results[name] = prepare_dataset(name, settings, args.limit, args.force)
        except Exception as e:
            print(f"[FAIL] Dataset '{name}' failed: {e}")
            results[name] = False

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    succeeded = sum(1 for ok in results.values() if ok)
    if succeeded == len(results):
        return 0
    elif succeeded > 0:
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
