"""GLM (Zhipu AI) Vision LLM implementation.

GLM-4V models (e.g. glm-4v-flash) accept OpenAI-style multimodal messages
(text + base64 image_url) at https://open.bigmodel.cn/api/paas/v4, so this
provider subclasses OpenAIVisionLLM and only overrides the default base URL.
"""

from __future__ import annotations

from typing import Any

from src.libs.llm.openai_vision_llm import OpenAIVisionLLM


class GlmVisionLLMError(RuntimeError):
    """Raised when GLM Vision API call fails."""


class GlmVisionLLM(OpenAIVisionLLM):
    """GLM (Zhipu AI) Vision LLM provider implementation.

    Supports glm-4v-flash (free tier) and other GLM-4V models for
    image captioning and visual question answering.

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> vision_llm = GlmVisionLLM(settings)
        >>> image = ImageInput(path="diagram.png")
        >>> response = vision_llm.chat_with_image("Describe this", image)
    """

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(self, settings: Any, **kwargs: Any) -> None:
        super().__init__(settings, **kwargs)
        # GLM-4V models cap max_tokens at 1024
        self.default_max_tokens = min(self.default_max_tokens, 1024)
