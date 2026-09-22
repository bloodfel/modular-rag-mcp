"""GLM (Zhipu AI) LLM implementation.

GLM provides an OpenAI-compatible API at https://open.bigmodel.cn/api/paas/v4,
so this provider subclasses OpenAILLM and only overrides the default base URL
and the environment variable fallback for the API key.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.libs.llm.openai_llm import OpenAILLM


class GlmLLMError(RuntimeError):
    """Raised when GLM API call fails."""


class GlmLLM(OpenAILLM):
    """GLM (Zhipu AI) LLM provider implementation.

    Uses the OpenAI-compatible chat completions protocol against the
    Zhipu AI open platform endpoint.

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = GlmLLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GLM LLM provider.

        Args:
            settings: Application settings containing LLM configuration.
            api_key: Optional API key override (falls back to settings.llm.api_key
                or the ZHIPUAI_API_KEY environment variable).
            base_url: Optional base URL override.
            **kwargs: Additional configuration overrides.
        """
        # OpenAILLM reads settings.llm.api_key, but GLM uses its own env var name
        if not api_key:
            api_key = os.environ.get("ZHIPUAI_API_KEY")
        super().__init__(settings, api_key=api_key, base_url=base_url, **kwargs)
