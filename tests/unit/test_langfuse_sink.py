"""Unit tests for the traces.jsonl → Langfuse mapping (observability.langfuse_sink)."""

from src.observability.langfuse_sink import trace_to_langfuse


def make_record(stages, finished="2026-09-21T10:00:02+00:00"):
    return {
        "trace_id": "t1",
        "trace_type": "query",
        "started_at": "2026-09-21T10:00:00+00:00",
        "finished_at": finished,
        "total_elapsed_ms": 2000,
        "stages": stages,
        "metadata": {"query": "test", "top_k": 5, "collection": "default", "source": "mcp"},
    }


class TestTraceToLangfuse:
    def test_trace_identity_and_input(self):
        out = trace_to_langfuse(make_record([]))
        assert out["trace"]["id"] == "t1"
        assert out["trace"]["name"] == "query"
        assert out["trace"]["input"] == "test"

    def test_one_span_per_stage(self):
        stages = [
            {"stage": "dense_retrieval", "timestamp": "2026-09-21T10:00:00.5+00:00", "data": {"result_count": 10}},
            {"stage": "rerank", "timestamp": "2026-09-21T10:00:01+00:00", "data": {"method": "jev"}},
        ]
        out = trace_to_langfuse(make_record(stages))
        assert [s["name"] for s in out["spans"]] == ["stage:dense_retrieval", "stage:rerank"]

    def test_span_uses_elapsed_ms(self):
        stages = [
            {"stage": "a", "timestamp": "2026-09-21T10:00:01+00:00", "elapsed_ms": 500, "data": {}},
            {"stage": "b", "timestamp": "2026-09-21T10:00:02+00:00", "elapsed_ms": 250, "data": {}},
        ]
        s1, s2 = trace_to_langfuse(make_record(stages))["spans"]
        assert s1["start_time"].isoformat().startswith("2026-09-21T10:00:00.5")
        assert s1["end_time"].isoformat().startswith("2026-09-21T10:00:01")
        assert s2["start_time"].isoformat().startswith("2026-09-21T10:00:01.75")

    def test_span_without_elapsed_chains_from_previous(self):
        stages = [
            {"stage": "a", "timestamp": "2026-09-21T10:00:01+00:00", "data": {}},
            {"stage": "b", "timestamp": "2026-09-21T10:00:02+00:00", "data": {}},
        ]
        s1, s2 = trace_to_langfuse(make_record(stages))["spans"]
        # legacy record: no elapsed_ms -> chain end-to-start
        assert s1["end_time"] == s2["start_time"]

    def test_span_output_summarizes_chunks(self):
        stages = [{
            "stage": "dense_retrieval",
            "timestamp": "2026-09-21T10:00:00+00:00",
            "elapsed_ms": 100,
            "data": {"method": "dense", "chunks": [{"id": 1}, {"id": 2}]},
        }]
        span = trace_to_langfuse(make_record(stages))["spans"][0]
        assert span["output"] == {"method": "dense", "chunk_count": 2}

    def test_stage_data_becomes_span_metadata(self):
        stages = [{"stage": "rerank", "timestamp": "2026-09-21T10:00:00+00:00", "data": {"method": "jev"}}]
        span = trace_to_langfuse(make_record(stages))["spans"][0]
        assert span["metadata"]["method"] == "jev"

    def test_trace_metadata_filtered_to_known_keys(self):
        out = trace_to_langfuse(make_record([]))
        assert "final_results" not in out["trace"]["metadata"]
        assert out["trace"]["metadata"]["top_k"] == 5


class TestTraceToOtlp:
    def test_otlp_shape_and_ids(self):
        from src.observability.langfuse_sink import trace_to_otlp, _hex_id

        stages = [{"stage": "rerank", "timestamp": "2026-09-21T10:00:01+00:00", "data": {"method": "jev"}}]
        body = trace_to_otlp(make_record(stages))
        spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
        root, child = spans[0], spans[1]
        assert len(root["traceId"]) == 32 and len(root["spanId"]) == 16
        assert root["name"] == "query" and "parentSpanId" not in root
        assert child["parentSpanId"] == root["spanId"]
        assert child["name"] == "stage:rerank"
        assert child["startTimeUnixNano"].isdigit() and child["endTimeUnixNano"].isdigit()
        assert _hex_id("550e8400-e29b-41d4-a716-446655440000") == "550e8400e29b41d4a716446655440000"

    def test_root_span_carries_session_and_input(self):
        from src.observability.langfuse_sink import trace_to_otlp

        body = trace_to_otlp(make_record([]))
        root = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attrs = {a["key"]: a["value"] for a in root["attributes"]}
        assert attrs["langfuse.session.id"]["stringValue"] == "default"
        assert attrs["langfuse.trace.input"]["stringValue"] == '"test"'
        assert attrs["langfuse.trace.metadata.top_k"]["intValue"] == "5"


def _otlp_spans(record):
    from src.observability.langfuse_sink import trace_to_otlp

    return trace_to_otlp(record)["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attrs(span):
    return {a["key"]: a["value"] for a in span["attributes"]}


def _llm_call_stage(
    purpose="rerank",
    model="glm-4-flash",
    tokens=(100, 20, 120),
    at="2026-09-21T10:00:01+00:00",
    label=None,
    prompt=None,
    response=None,
):
    data = {
        "purpose": purpose,
        "model": model,
        "tokens_in": tokens[0],
        "tokens_out": tokens[1],
        "total_tokens": tokens[2],
        "usage_reported": True,
    }
    if label:
        data["label"] = label
    if prompt:
        data["prompt"] = prompt
    if response:
        data["response"] = response
    return {"stage": "llm_call", "timestamp": at, "elapsed_ms": 500, "data": data}


class TestObservationTypes:
    """Langfuse asks for the most specific observation type, not a generic span."""

    def test_retrieval_stages_are_retrievers(self):
        stages = [
            {"stage": "dense_retrieval", "timestamp": "2026-09-21T10:00:00+00:00", "data": {}},
            {"stage": "sparse_retrieval", "timestamp": "2026-09-21T10:00:00.1+00:00", "data": {}},
        ]
        types = [_attrs(s)["langfuse.observation.type"]["stringValue"] for s in _otlp_spans(make_record(stages))[1:]]
        assert types == ["retriever", "retriever"]

    def test_embedding_stage_is_embedding(self):
        stages = [{"stage": "embed", "timestamp": "2026-09-21T10:00:00+00:00", "data": {}}]
        assert _attrs(_otlp_spans(make_record(stages))[1])["langfuse.observation.type"]["stringValue"] == "embedding"

    def test_unmapped_stage_is_a_plain_span(self):
        stages = [{"stage": "fusion", "timestamp": "2026-09-21T10:00:00+00:00", "data": {}}]
        assert _attrs(_otlp_spans(make_record(stages))[1])["langfuse.observation.type"]["stringValue"] == "span"


class TestGenerationObservations:
    """An llm_call must reach Langfuse as a priced generation."""

    def test_llm_call_is_a_generation(self):
        spans = _otlp_spans(make_record([_llm_call_stage()]))
        assert _attrs(spans[1])["langfuse.observation.type"]["stringValue"] == "generation"

    def test_model_name_is_sent(self):
        spans = _otlp_spans(make_record([_llm_call_stage(model="glm-4-flash")]))
        assert _attrs(spans[1])["langfuse.observation.model.name"]["stringValue"] == "glm-4-flash"

    def test_usage_details_are_sent_as_json(self):
        import json

        spans = _otlp_spans(make_record([_llm_call_stage(tokens=(100, 20, 120))]))
        usage = json.loads(_attrs(spans[1])["langfuse.observation.usage_details"]["stringValue"])
        assert usage == {"input": 100, "output": 20, "total": 120}

    def test_unknown_usage_omits_usage_details(self):
        stage = _llm_call_stage()
        stage["data"].update({"tokens_in": None, "tokens_out": None, "total_tokens": None})
        spans = _otlp_spans(make_record([stage]))
        assert "langfuse.observation.usage_details" not in _attrs(spans[1])
        # the model is still reported, so the call stays identifiable
        assert "langfuse.observation.model.name" in _attrs(spans[1])

    def test_generation_nests_under_the_stage_that_caused_it(self):
        stages = [
            {"stage": "rerank", "timestamp": "2026-09-21T10:00:02+00:00", "data": {"method": "llm"}},
            _llm_call_stage(purpose="rerank"),
        ]
        root, rerank, generation = _otlp_spans(make_record(stages))
        assert rerank["parentSpanId"] == root["spanId"]
        assert generation["parentSpanId"] == rerank["spanId"]

    def test_unmatched_purpose_falls_back_to_the_root(self):
        spans = _otlp_spans(make_record([_llm_call_stage(purpose="no-such-stage")]))
        assert spans[1]["parentSpanId"] == spans[0]["spanId"]

    def test_repeated_llm_calls_get_distinct_span_ids(self):
        """Every llm_call shares a name, so ids must not be keyed by name."""
        stages = [
            _llm_call_stage(at="2026-09-21T10:00:01+00:00"),
            _llm_call_stage(at="2026-09-21T10:00:02+00:00"),
            _llm_call_stage(at="2026-09-21T10:00:03+00:00"),
        ]
        ids = [s["spanId"] for s in _otlp_spans(make_record(stages))]
        assert len(ids) == len(set(ids)) == 4  # 3 generations + the root


class TestCallIdentity:
    """A dozen calls from one stage must be tellable apart in the waterfall."""

    def test_each_call_gets_a_numbered_name(self):
        stages = [
            _llm_call_stage(purpose="chunk_refiner", at="2026-09-21T10:00:01+00:00"),
            _llm_call_stage(purpose="chunk_refiner", at="2026-09-21T10:00:02+00:00"),
            _llm_call_stage(purpose="metadata_enricher", at="2026-09-21T10:00:03+00:00"),
        ]
        names = [s["name"] for s in _otlp_spans(make_record(stages))[1:]]
        assert names == ["chunk_refiner#1", "chunk_refiner#2", "metadata_enricher#1"]

    def test_prompt_is_sent_as_input(self):
        spans = _otlp_spans(make_record([_llm_call_stage(prompt="refine this")]))
        assert _attrs(spans[1])["langfuse.observation.input"]["stringValue"] == '"refine this"'

    def test_response_is_sent_as_output(self):
        spans = _otlp_spans(make_record([_llm_call_stage(response="refined")]))
        assert _attrs(spans[1])["langfuse.observation.output"]["stringValue"] == '"refined"'

    def test_prompt_is_not_repeated_in_metadata(self):
        """It is already the observation input; duplicating it bloats the panel."""
        spans = _otlp_spans(make_record([_llm_call_stage(prompt="refine this", response="done")]))
        keys = _attrs(spans[1])
        assert "langfuse.observation.metadata.prompt" not in keys
        assert "langfuse.observation.metadata.response" not in keys

    def test_label_reaches_metadata(self):
        spans = _otlp_spans(make_record([_llm_call_stage(label="chunk_0007")]))
        assert _attrs(spans[1])["langfuse.observation.metadata.label"]["stringValue"] == "chunk_0007"

    def test_call_without_prompt_still_exports(self):
        """Traces recorded before the prompt was captured must not break."""
        spans = _otlp_spans(make_record([_llm_call_stage()]))
        attrs = _attrs(spans[1])
        assert "langfuse.observation.input" not in attrs
        assert attrs["langfuse.observation.model.name"]["stringValue"] == "glm-4-flash"


class TestStableIds:
    """Langfuse keys observations on the span id, so a repeat export must
    overwrite rather than pile up duplicates."""

    def test_span_ids_are_identical_across_exports(self):
        record = make_record([_llm_call_stage(), _llm_call_stage(at="2026-09-21T10:00:02+00:00")])
        first = [s["spanId"] for s in _otlp_spans(record)]
        second = [s["spanId"] for s in _otlp_spans(record)]
        assert first == second
        assert len(set(first)) == len(first)

    def test_different_traces_get_different_span_ids(self):
        a = [s["spanId"] for s in _otlp_spans(make_record([_llm_call_stage()]))]
        rec_b = make_record([_llm_call_stage()])
        rec_b["trace_id"] = "t2"
        b = [s["spanId"] for s in _otlp_spans(rec_b)]
        assert a != b


class TestSendScores:
    """Evaluation scores travel to Langfuse over the v3 scores API."""

    def test_writes_one_score_per_metric_under_the_coerced_trace_id(self, monkeypatch) -> None:
        from src.observability import langfuse_sink

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setattr(langfuse_sink, "_credentials", lambda: ("https://lf.example", "pk", "sk"))

        calls = []

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

        def fake_post(url, auth=None, timeout=None, json=None):
            calls.append((url, auth, json))
            return FakeResponse()

        import httpx
        monkeypatch.setattr(httpx, "post", fake_post)

        local_id = "c550b25a-1a2b-4c3d-9e8f-000011112222"
        written = langfuse_sink.send_scores(local_id, {"faithfulness": 1.0, "context_recall": 0.5})

        assert written == 2
        assert all(url.endswith("/api/public/scores") for url, _, _ in calls)
        assert [c[2]["name"] for c in calls] == ["ragas.faithfulness", "ragas.context_recall"]
        # dashes are stripped: Langfuse knows the trace by its 32-hex id
        assert all(c[2]["traceId"] == local_id.replace("-", "") for c in calls)
        assert all(c[2]["dataType"] == "NUMERIC" for c in calls)

    def test_noop_without_credentials(self, monkeypatch) -> None:
        from src.observability import langfuse_sink

        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        assert langfuse_sink.send_scores("t", {"faithfulness": 1.0}) == 0
