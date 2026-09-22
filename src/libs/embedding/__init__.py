"""
Embedding Module.

This package contains embedding service abstractions and implementations:
- Base embedding class
- Embedding factory
- Provider implementations (OpenAI, Azure, Ollama)
"""

from src.libs.embedding.azure_embedding import AzureEmbedding
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.ollama_embedding import OllamaEmbedding
from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.glm_embedding import GlmEmbedding

# Register GLM embedding provider with factory
EmbeddingFactory.register_provider("glm", GlmEmbedding)

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "OpenAIEmbedding",
    "AzureEmbedding",
    "OllamaEmbedding",
    "GlmEmbedding",
]
