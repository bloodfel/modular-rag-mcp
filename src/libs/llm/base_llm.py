"""Abstract base class for LLM providers.

This module defines the pluggable interface for Language Model providers,
enabling seamless switching between different backends (OpenAI, Azure, Ollama, etc.)
through configuration-driven instantiation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.trace.trace_context import TraceContext


@dataclass
class Message:
    """Represents a single message in a chat conversation.
    
    Attributes:
        role: The role of the message sender ('system', 'user', or 'assistant').
        content: The text content of the message.
    """
    role: str
    content: str


@dataclass
class ChatResponse:
    """Response from an LLM chat completion.
    
    Attributes:
        content: The generated text response.
        model: The model identifier that generated the response.
        usage: Optional token usage statistics (prompt_tokens, completion_tokens, total_tokens).
        raw_response: Optional raw response from the provider for debugging.
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


# Ceiling on how much prompt/response text a trace carries. Enough to
# recognise the call from the trace alone; the full text stays in the pipeline.
_PREVIEW_CHARS = 4000


def _preview(text: str) -> str:
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[:_PREVIEW_CHARS] + f"…[{len(text) - _PREVIEW_CHARS} chars truncated]"


def _conversation_preview(messages: List[Message]) -> str:
    """Flatten the messages actually sent, so a trace shows what was asked."""
    if len(messages) == 1:
        return _preview(messages[0].content)
    return "\n\n".join(f"[{m.role}]\n{_preview(m.content)}" for m in messages)


def record_llm_usage(
    trace: Optional[TraceContext],
    response: ChatResponse,
    purpose: Optional[str] = None,
    label: Optional[str] = None,
    elapsed_ms: Optional[float] = None,
    prompt: Optional[str] = None,
) -> None:
    """Record one LLM call on *trace* as an ``llm_call`` stage.

    Args:
        trace: Optional TraceContext; no-op when None.
        response: The response returned by the provider.
        purpose: Which stage made the call (e.g. ``"rerank"``). The Langfuse
            exporter nests the generation under the stage of the same name.
        label: What this particular call was about (e.g. a chunk id), so the
            twelve calls of one stage can be told apart.
        elapsed_ms: Measured call duration.
        prompt: The conversation sent, truncated by :data:`_PREVIEW_CHARS`.
    """
    if trace is None:
        return

    usage = response.usage or {}
    data: Dict[str, Any] = {
        "purpose": purpose or "llm",
        "model": response.model,
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "usage_reported": bool(usage),
    }
    if label:
        data["label"] = label
    if prompt:
        data["prompt"] = prompt
    if response.content:
        data["response"] = _preview(response.content)

    trace.record_stage("llm_call", data, elapsed_ms=elapsed_ms)


class BaseLLM(ABC):
    """Abstract base class for LLM providers.
    
    All LLM implementations must inherit from this class and implement
    the _chat() method. This ensures consistent interface across different
    providers (OpenAI, Azure, DeepSeek, Ollama, etc.).

    Design Principles Applied:
    - Pluggable: Subclasses can be swapped without changing upstream code.
    - Observable: chat() times every call and records model, token usage and
      latency on the TraceContext, so a new provider inherits that for free.
    - Config-Driven: Instances are created via factory based on settings.
    """

    def chat(
        self,
        messages: List[Message],
        trace: Optional[TraceContext] = None,
        purpose: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion response, recording the call on *trace*.

        This is the instrumented entry point shared by every provider; the
        provider-specific work lives in :meth:`_chat`. Validation happens here
        so no provider can skip it.

        Args:
            messages: List of conversation messages (role + content).
            trace: Optional TraceContext; when given, the call is recorded as
                an ``llm_call`` stage carrying model, tokens, latency and the
                prompt, so the call can be identified from the trace alone.
            purpose: Which stage triggered the call (e.g. "rerank"). The
                Langfuse exporter nests the generation under that stage.
            label: What this call was about (e.g. a chunk id). A stage that
                makes many calls needs this to tell them apart.
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.).

        Returns:
            ChatResponse containing the generated text and metadata.

        Raises:
            ValueError: If messages list is empty or malformed.
            RuntimeError: If the LLM provider call fails.
        """
        self.validate_messages(messages)
        started = time.monotonic()
        response = self._chat(messages, **kwargs)
        record_llm_usage(
            trace,
            response,
            purpose=purpose,
            label=label,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            prompt=_conversation_preview(messages),
        )
        return response

    @abstractmethod
    def _chat(self, messages: List[Message], **kwargs: Any) -> ChatResponse:
        """Issue the provider-specific completion request.

        Call through :meth:`chat` rather than directly, otherwise the call is
        invisible to tracing.

        Args:
            messages: Pre-validated conversation messages.
            **kwargs: Provider-specific parameters.

        Returns:
            ChatResponse containing the generated text and metadata.
        """
        pass

    def validate_messages(self, messages: List[Message]) -> None:
        """Validate message list structure.

        Args:
            messages: List of messages to validate.

        Raises:
            ValueError: If messages list is empty or contains invalid roles.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        valid_roles = {"system", "user", "assistant"}
        for i, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise ValueError(f"Message at index {i} is not a Message instance")
            if msg.role not in valid_roles:
                raise ValueError(
                    f"Message at index {i} has invalid role '{msg.role}'. "
                    f"Must be one of: {valid_roles}"
                )
            if not msg.content or not msg.content.strip():
                raise ValueError(f"Message at index {i} has empty content")
