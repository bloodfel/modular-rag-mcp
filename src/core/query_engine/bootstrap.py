"""Shared query-component assembly.

Single wiring point for the retrieval stack (vector store -> dense/sparse
retrievers -> hybrid search + reranker).  Used by the CLI query script,
the MCP ``query_knowledge_hub`` tool and the dashboard Query Playground so
the BM25 index layout and retriever wiring cannot drift between entry
points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from src.core.settings import resolve_path

if TYPE_CHECKING:
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import CoreReranker
    from src.core.settings import Settings


def build_query_components(
    settings: "Settings", collection: str
) -> Tuple["HybridSearch", "CoreReranker"]:
    """Build HybridSearch + CoreReranker bound to *collection*.

    Args:
        settings: Application settings.
        collection: Target collection name; also selects the per-collection
            BM25 index directory (``data/db/bm25/<collection>``) — the same
            layout every query entry point reads.

    Returns:
        Tuple of (hybrid_search, reranker).
    """
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.core.query_engine.reranker import create_core_reranker
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    vector_store = VectorStoreFactory.create(
        settings,
        collection_name=collection,
    )
    embedding_client = EmbeddingFactory.create(settings)
    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection
    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )
    reranker = create_core_reranker(settings=settings)
    return hybrid_search, reranker
