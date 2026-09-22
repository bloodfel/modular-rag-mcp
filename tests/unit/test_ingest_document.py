"""Tests for the ingest_document MCP tool."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from src.mcp_server.tools.ingest_document import (
    IngestDocumentTool,
    sanitize_doc_name,
)


@pytest.fixture
def mock_settings():
    return MagicMock()


@contextlib.contextmanager
def ingest_patches(n_chunks: int = 2, stale: int = 3):
    """Patch every ingestion component used by IngestDocumentTool.ingest.

    Yields a dict of the fake components so tests can assert on calls.
    """
    chunks = [MagicMock(id=f"doc_{i:04d}_0000_hash{i}", text=f"chunk {i}") for i in range(n_chunks)]

    chunker = MagicMock()
    chunker.split_document.return_value = chunks
    embedding_factory = MagicMock()
    embedding_factory.create.return_value = MagicMock()
    dense = MagicMock()
    dense.encode.return_value = [[0.1, 0.2]] * n_chunks
    upserter = MagicMock()
    upserter.upsert.return_value = [c.id for c in chunks]
    sparse = MagicMock()
    sparse.encode.return_value = [{"term": "x", "count": 1}] * n_chunks
    store = MagicMock()
    store.delete_by_metadata.return_value = stale
    indexer = MagicMock()
    document_cls = MagicMock()

    specs = {
        "src.core.types": {"Document": document_cls},
        "src.ingestion.chunking.document_chunker": {"DocumentChunker": MagicMock(return_value=chunker)},
        "src.ingestion.embedding.dense_encoder": {"DenseEncoder": MagicMock(return_value=dense)},
        "src.ingestion.embedding.sparse_encoder": {"SparseEncoder": MagicMock(return_value=sparse)},
        "src.ingestion.storage.vector_upserter": {"VectorUpserter": MagicMock(return_value=upserter)},
        "src.ingestion.storage.bm25_indexer": {"BM25Indexer": MagicMock(return_value=indexer)},
        "src.libs.vector_store.chroma_store": {"ChromaStore": MagicMock(return_value=store)},
        "src.libs.embedding.embedding_factory": {"EmbeddingFactory": embedding_factory},
    }
    with contextlib.ExitStack() as stack:
        for module_name, attrs in specs.items():
            stack.enter_context(patch.multiple(module_name, **attrs))
        yield {
            "chunks": chunks,
            "chunker": chunker,
            "dense": dense,
            "upserter": upserter,
            "sparse": sparse,
            "store": store,
            "indexer": indexer,
            "document_cls": document_cls,
        }


class TestSanitizeDocName:
    def test_strips_unsafe_chars(self):
        assert sanitize_doc_name("my doc/v1?") == "my-doc-v1"

    def test_keeps_cjk(self):
        assert sanitize_doc_name("产品手册") == "产品手册"

    def test_empty_becomes_untitled(self):
        assert sanitize_doc_name("///") == "untitled"


class TestIngestDocumentTool:
    @pytest.mark.asyncio
    async def test_rejects_empty_name_and_content(self, mock_settings):
        tool = IngestDocumentTool(settings=mock_settings)
        result = await tool.execute(name="  ", content="hello")
        assert result.isError
        result = await tool.execute(name="doc", content="")
        assert result.isError

    @pytest.mark.asyncio
    async def test_successful_ingest_replaces_stale_version(self, mock_settings):
        tool = IngestDocumentTool(settings=mock_settings)
        with ingest_patches(n_chunks=2, stale=3):
            result = await tool.execute(name="handbook", content="a" * 500)

        assert not result.isError
        assert "2 chunk(s)" in result.content[0].text
        assert "3 stale chunk(s)" in result.content[0].text

    @pytest.mark.asyncio
    async def test_chroma_cleanup_targets_exact_source_path(self, mock_settings):
        tool = IngestDocumentTool(settings=mock_settings)
        with ingest_patches() as mocks:
            await tool.execute(name="handbook", content="a" * 500)
        mocks["store"].delete_by_metadata.assert_called_once_with(
            {"source_path": "mcp://default/handbook"}
        )

    @pytest.mark.asyncio
    async def test_bm25_reingest_passes_doc_id_prefix(self, mock_settings):
        tool = IngestDocumentTool(settings=mock_settings)
        with ingest_patches() as mocks:
            await tool.execute(name="handbook", content="a" * 500)
        _, kwargs = mocks["indexer"].add_documents.call_args
        assert kwargs["doc_id"] == "mcp-default-handbook"

    @pytest.mark.asyncio
    async def test_empty_chunks_reports_gracefully(self, mock_settings):
        tool = IngestDocumentTool(settings=mock_settings)
        with ingest_patches() as mocks:
            mocks["chunker"].split_document.return_value = []
            result = await tool.execute(name="blank", content="a" * 100)
        assert not result.isError
        assert "no chunks" in result.content[0].text

    @pytest.mark.asyncio
    async def test_pipeline_failure_returns_tool_error(self, mock_settings):
        tool = IngestDocumentTool(settings=mock_settings)
        with ingest_patches() as mocks:
            mocks["upserter"].upsert.side_effect = RuntimeError("boom")
            result = await tool.execute(name="bad", content="a" * 100)
        assert result.isError
        assert "boom" in result.content[0].text
