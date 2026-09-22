"""Tests for the BaseLLM instrumentation hook.

Every provider is expected to implement ``_chat``; ``chat`` times the call and
records model, token usage and latency on the trace. These tests pin that
contract so a provider cannot quietly bypass observability.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from src.core.trace.trace_context import TraceContext
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class StubLLM(BaseLLM):
    """Provider stub implementing only the ``_chat`` contract."""

    def __init__(
        self,
        content: str = "ok",
        model: str = "stub-model",
        usage: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.model = model
        self.usage = (
            {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            if usage is None
            else usage
        )
        self.error = error
        self.calls: List[List[Message]] = []
        self.kwargs_seen: List[dict] = []

    def _chat(self, messages: List[Message], **kwargs: Any) -> ChatResponse:
        self.calls.append(messages)
        self.kwargs_seen.append(kwargs)
        if self.error:
            raise self.error
        return ChatResponse(content=self.content, model=self.model, usage=self.usage)


def _llm_calls(trace: TraceContext) -> list:
    return [s for s in trace.stages if s["stage"] == "llm_call"]


class TestRecording:
    def test_records_model_and_tokens(self) -> None:
        trace = TraceContext()

        StubLLM().chat([Message(role="user", content="hi")], trace=trace)

        calls = _llm_calls(trace)
        assert len(calls) == 1
        data = calls[0]["data"]
        assert data["model"] == "stub-model"
        assert data["tokens_in"] == 11
        assert data["tokens_out"] == 7
        assert data["total_tokens"] == 18
        assert data["usage_reported"] is True

    def test_records_elapsed_time(self) -> None:
        trace = TraceContext()

        StubLLM().chat([Message(role="user", content="hi")], trace=trace)

        assert _llm_calls(trace)[0]["elapsed_ms"] >= 0.0

    def test_purpose_labels_the_call(self) -> None:
        trace = TraceContext()

        StubLLM().chat([Message(role="user", content="hi")], trace=trace, purpose="rerank")

        assert _llm_calls(trace)[0]["data"]["purpose"] == "rerank"

    def test_purpose_defaults_to_llm(self) -> None:
        trace = TraceContext()

        StubLLM().chat([Message(role="user", content="hi")], trace=trace)

        assert _llm_calls(trace)[0]["data"]["purpose"] == "llm"

    def test_no_trace_is_silent(self) -> None:
        """Tracing is optional; an untraced call must still work."""
        response = StubLLM(content="hello").chat([Message(role="user", content="hi")])

        assert response.content == "hello"

    def test_missing_usage_is_flagged_not_invented(self) -> None:
        trace = TraceContext()

        StubLLM(usage={}).chat([Message(role="user", content="hi")], trace=trace)

        data = _llm_calls(trace)[0]["data"]
        assert data["usage_reported"] is False
        assert data["tokens_in"] is None
        assert data["total_tokens"] is None


class TestCallIdentity:
    """A stage can make a dozen calls; the trace must say which was which."""

    def test_prompt_is_recorded(self) -> None:
        trace = TraceContext()

        StubLLM().chat([Message(role="user", content="refine this chunk")], trace=trace)

        assert _llm_calls(trace)[0]["data"]["prompt"] == "refine this chunk"

    def test_multi_message_conversation_keeps_roles(self) -> None:
        trace = TraceContext()

        StubLLM().chat(
            [
                Message(role="system", content="be terse"),
                Message(role="user", content="hello"),
            ],
            trace=trace,
        )

        prompt = _llm_calls(trace)[0]["data"]["prompt"]
        assert "[system]\nbe terse" in prompt
        assert "[user]\nhello" in prompt

    def test_response_is_recorded(self) -> None:
        trace = TraceContext()

        StubLLM(content="refined text").chat(
            [Message(role="user", content="hi")], trace=trace
        )

        assert _llm_calls(trace)[0]["data"]["response"] == "refined text"

    def test_label_says_what_the_call_was_about(self) -> None:
        trace = TraceContext()

        StubLLM().chat(
            [Message(role="user", content="hi")], trace=trace, label="chunk_0007"
        )

        assert _llm_calls(trace)[0]["data"]["label"] == "chunk_0007"

    def test_label_is_absent_when_not_given(self) -> None:
        trace = TraceContext()

        StubLLM().chat([Message(role="user", content="hi")], trace=trace)

        assert "label" not in _llm_calls(trace)[0]["data"]

    def test_long_prompt_and_response_are_truncated(self) -> None:
        """Traces stay readable and bounded even when a chunk is huge."""
        from src.libs.llm.base_llm import _PREVIEW_CHARS

        trace = TraceContext()
        StubLLM(content="x" * (_PREVIEW_CHARS + 500)).chat(
            [Message(role="user", content="y" * (_PREVIEW_CHARS + 500))], trace=trace
        )

        data = _llm_calls(trace)[0]["data"]
        assert len(data["prompt"]) < _PREVIEW_CHARS + 100
        assert "chars truncated" in data["prompt"]
        assert "chars truncated" in data["response"]


class TestContract:
    def test_validation_still_applies_to_every_provider(self) -> None:
        """The template validates, so no provider can skip it."""
        with pytest.raises(ValueError, match="cannot be empty"):
            StubLLM().chat([])

    def test_invalid_role_rejected_through_chat(self) -> None:
        with pytest.raises(ValueError, match="invalid role"):
            StubLLM().chat([Message(role="narrator", content="hi")])

    def test_failed_call_records_nothing_and_propagates(self) -> None:
        trace = TraceContext()

        with pytest.raises(RuntimeError, match="provider down"):
            StubLLM(error=RuntimeError("provider down")).chat(
                [Message(role="user", content="hi")], trace=trace
            )

        assert _llm_calls(trace) == []

    def test_purpose_is_not_forwarded_to_the_provider(self) -> None:
        """purpose is a tracing label, not a provider parameter."""
        llm = StubLLM()

        llm.chat([Message(role="user", content="hi")], purpose="rerank")

        assert "purpose" not in llm.kwargs_seen[0]

    def test_provider_kwargs_are_forwarded(self) -> None:
        llm = StubLLM()

        llm.chat([Message(role="user", content="hi")], temperature=0.3)

        assert llm.kwargs_seen[0] == {"temperature": 0.3}
