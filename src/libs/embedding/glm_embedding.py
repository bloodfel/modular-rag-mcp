"""GLM (Zhipu AI) Embedding implementation.

GLM provides an OpenAI-compatible embeddings API at
https://open.bigmodel.cn/api/paas/v4 (model: embedding-3), so this provider
subclasses OpenAIEmbedding and overrides the base URL plus the
dimensions-parameter rule (GLM models are named "embedding-*", not
"text-embedding-3-*").
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.libs.embedding.openai_embedding import OpenAIEmbedding, OpenAIEmbeddingError


class GlmEmbedding(OpenAIEmbedding):
    """GLM (Zhipu AI) Embedding provider implementation.

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = GlmEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts using the GLM API.

        Args:
            texts: List of text strings to embed. Must not be empty.
            trace: Optional TraceContext for observability.
            **kwargs: Override parameters (dimensions, etc.).

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            OpenAIEmbeddingError: If API call fails.
        """
        # Validate input
        self.validate_texts(texts)

        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Python package not installed. "
                "Install with: pip install openai"
            ) from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        api_params: dict = {
            "input": texts,
            "model": self.model,
        }
        # GLM embedding-3 supports 256/512/1024/2048 dimensions
        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None and self.model.startswith("embedding-"):
            api_params["dimensions"] = dimensions

        try:
            response = client.embeddings.create(**api_params)
        except Exception as e:
            raise OpenAIEmbeddingError(
                f"GLM Embeddings API call failed: {e}"
            ) from e

        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as e:
            raise OpenAIEmbeddingError(
                f"Failed to parse GLM Embeddings API response: {e}"
            ) from e

        if len(embeddings) != len(texts):
            raise OpenAIEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        return embeddings

    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension for the configured GLM model."""
        if self.dimensions is not None:
            return self.dimensions
        # embedding-3 default output dimension
        return 2048
