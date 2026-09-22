"""Send finished traces to Langfuse.

Maps a TraceContext record to one Langfuse trace with one span per pipeline
stage, so a query's full retrieval path (bm25/dense/fusion/rerank) and an
ingestion's LLM calls are visible in the Langfuse UI.

``TraceCollector`` calls :func:`send_trace` for every trace it persists, so
reporting is not something a caller opts into. Sending is best-effort: a
missing key or an unreachable server never fails the work being traced.

Credentials come from .env / environment:
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

Backfilling already-written traces lives in scripts/export_langfuse.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_URL, uuid5

from src.core.settings import _load_dotenv_file, resolve_path

logger = logging.getLogger(__name__)

# Load repo-root .env (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST)
# so users don't need to export variables manually.
_load_dotenv_file()

TRACES_PATH = resolve_path("logs/traces.jsonl")


def _parse_ts(value: str) -> datetime:
    """Parse an ISO timestamp, tolerating a missing timezone."""
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def trace_to_langfuse(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map one TraceContext JSON record to Langfuse trace/span payloads.

    Returns a plain dict (no SDK import needed) so the mapping is unit-testable:
        {"trace": {...}, "spans": [{...}, ...]}
    """
    started = _parse_ts(record["started_at"])
    finished = _parse_ts(record.get("finished_at") or record["started_at"])
    stages = record.get("stages", [])

    spans = []
    prev_end = started
    for index, stage in enumerate(stages):
        stage_end = _parse_ts(stage["timestamp"])
        elapsed = stage.get("elapsed_ms")
        if isinstance(elapsed, (int, float)):
            stage_start = stage_end - timedelta(milliseconds=float(elapsed))
        else:
            # Legacy record without elapsed_ms: chain from the previous stage
            stage_start = prev_end if prev_end <= stage_end else stage_end
        prev_end = stage_end
        data = stage.get("data") or {}
        # Compact output for the span detail panel (full data stays in metadata)
        summary = {k: v for k, v in data.items() if k != "chunks"}
        if "chunks" in data:
            summary["chunk_count"] = len(data["chunks"])
        spans.append(
            {
                "name": f"stage:{stage['stage']}",
                "start_time": stage_start,
                "end_time": max(stage_start, stage_end),
                "metadata": data,
                "output": summary or None,
            }
        )

    metadata = record.get("metadata", {})
    return {
        "trace": {
            "id": record["trace_id"],
            "name": record.get("trace_type", "query"),
            "timestamp": started,
            "input": metadata.get("query"),
            "metadata": {
                key: metadata[key]
                for key in ("top_k", "collection", "source")
                if key in metadata
            },
        },
        "spans": [
            {
                "name": span["name"],
                "start_time": span["start_time"],
                "end_time": span["end_time"] or finished,
                "metadata": span["metadata"],
                "output": span.get("output"),
            }
            for span in spans
        ],
    }


def export(traces_path: Path = TRACES_PATH, last: int = 10, trace_type: Optional[str] = None) -> int:
    """Export the most recent traces to Langfuse. Returns count exported.

    Uses OTLP (OpenTelemetry HTTP/JSON) ingestion — Langfuse Cloud rejects
    the legacy ingestion API for organizations created after 2026-09-16,
    and OTLP spans carry explicit unix-nano timestamps so historical
    backfill works.
    """
    if not traces_path.exists():
        print(f"[skip] no trace file at {traces_path}")
        return 0

    records = []
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if trace_type:
        records = [r for r in records if r.get("trace_type") == trace_type]
    records = records[-last:]

    exported = 0
    for record in records:
        if _post_otlp(trace_to_otlp(record)):
            exported += 1
    print(f"[ok] exported {exported}/{len(records)} trace(s) to {os.environ.get('LANGFUSE_HOST', 'Langfuse')}")
    return exported


def send_trace(record: Dict[str, Any]) -> bool:
    """Send one trace record to Langfuse. Returns True on success.

    Best-effort by contract: tracing must never fail the work being traced,
    so every failure is logged and swallowed. The caller (TraceCollector)
    has already persisted the trace locally by the time this runs.

    Logging goes through the logging module rather than stdout, because the
    MCP stdio server reserves stdout for JSON-RPC.
    """
    if not langfuse_configured():
        return False
    try:
        return _post_otlp(trace_to_otlp(record))
    except Exception as e:  # noqa: BLE001 - any failure is non-fatal here
        logger.warning("Langfuse export failed, trace kept locally only: %s", e)
        return False


def langfuse_configured() -> bool:
    """True when LANGFUSE_PUBLIC_KEY/SECRET_KEY are present."""
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def send_scores(
    trace_id: str,
    scores: Dict[str, float],
    *,
    prefix: str = "ragas.",
) -> int:
    """Attach evaluation scores to an already-reported trace.

    Same best-effort contract as send_trace: a failed score write must never
    fail the evaluation run that produced it. Returns how many scores were
    written. ``trace_id`` is the local trace id; it is coerced with
    ``_hex_id`` into the 32-hex id Langfuse knows the trace by.
    """
    if not scores or not langfuse_configured():
        return 0

    import httpx

    host, public_key, secret_key = _credentials()
    langfuse_trace_id = _hex_id(trace_id)
    written = 0

    for name, value in scores.items():
        if value is None:
            continue
        try:
            response = httpx.post(
                f"{host}/api/public/scores",
                auth=(public_key, secret_key),
                timeout=30,
                json={
                    "traceId": langfuse_trace_id,
                    "name": f"{prefix}{name}",
                    "value": float(value),
                    "dataType": "NUMERIC",
                },
            )
            response.raise_for_status()
            written += 1
        except Exception as e:  # noqa: BLE001 - score write is non-fatal
            logger.warning(
                "Langfuse score write failed for %s (%s): %s", trace_id, name, e
            )

    return written


def _nanos(dt: datetime) -> str:
    """Datetime -> unix nanoseconds string (OTLP timestamp format)."""
    return str(int(dt.timestamp() * 1_000_000_000))


def _hex_id(value: str) -> str:
    """Coerce an arbitrary id into a 32-char hex OTLP trace id."""
    cleaned = "".join(c for c in value.lower() if c in "0123456789abcdef")
    if len(cleaned) == 32:
        return cleaned
    return uuid5(NAMESPACE_URL, value).hex


def _stable_span_id(trace_id: str, discriminator: Any) -> str:
    """Deterministic 16-hex span id derived from the trace id.

    Langfuse keys observations on the span id, so a stable id makes a repeat
    export of the same trace an update rather than a duplicate.
    """
    return uuid5(NAMESPACE_URL, f"{trace_id}:{discriminator}").hex[:16]


def _kv(key: str, value: Any) -> Dict[str, Any]:
    """Build an OTLP attribute; objects serialize as jsonValue strings."""
    if isinstance(value, bool):
        typed = {"boolValue": value}
    elif isinstance(value, int):
        typed = {"intValue": str(value)}
    elif isinstance(value, float):
        typed = {"doubleValue": value}
    elif isinstance(value, (dict, list)):
        typed = {"kvlistValue": {
            "values": [_kv(k, v) for k, v in value.items()]
            if isinstance(value, dict) else []
        }} if isinstance(value, dict) else {"stringValue": json.dumps(value, ensure_ascii=False, default=str)}
    else:
        typed = {"stringValue": str(value)}
    return {"key": key, "value": typed}


def _scalar(value: Any) -> Any:
    """Scalars pass through; containers become compact JSON strings."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_attr(key: str, value: Any) -> Dict[str, Any]:
    """Attribute whose value is a JSON string (Langfuse input/output convention)."""
    return {"key": key, "value": {"stringValue": json.dumps(value, ensure_ascii=False, default=str)}}


# Langfuse asks for the most specific observation type available; anything not
# listed is a plain span.
_OBSERVATION_TYPES = {
    "dense_retrieval": "retriever",
    "sparse_retrieval": "retriever",
    "embed": "embedding",
    "llm_call": "generation",
}
_DEFAULT_OBSERVATION_TYPE = "span"


def _observation_type(stage_name: str) -> str:
    return _OBSERVATION_TYPES.get(stage_name, _DEFAULT_OBSERVATION_TYPE)


def _generation_attrs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Model + token attributes for a generation.

    Langfuse prices the call itself from its own model table once it knows the
    model name and the token counts, so no price list is kept here.
    """
    attrs: List[Dict[str, Any]] = []
    if data.get("model"):
        attrs.append(_kv("langfuse.observation.model.name", data["model"]))
    usage = {
        key: value
        for key, value in (
            ("input", data.get("tokens_in")),
            ("output", data.get("tokens_out")),
            ("total", data.get("total_tokens")),
        )
        if value is not None
    }
    if usage:
        attrs.append(_json_attr("langfuse.observation.usage_details", usage))
    return attrs


def trace_to_otlp(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map one TraceContext record to an OTLP traces export body.

    The root span represents the trace (its langfuse.* attributes populate
    trace-level fields); one child span per pipeline stage carries the
    stage data as metadata.
    """
    mapped = trace_to_langfuse(record)
    trace = mapped["trace"]
    trace_id = _hex_id(trace["id"])
    # Ids derive from the trace id rather than random bytes: re-exporting the
    # same trace then updates its observations instead of duplicating them.
    root_span_id = _stable_span_id(trace["id"], "root")

    def span(span_id: str, name: str, start: datetime, end: datetime,
             parent: Optional[str], attrs: List[Dict[str, Any]]) -> Dict[str, Any]:
        s: Dict[str, Any] = {
            "traceId": trace_id,
            "spanId": span_id,
            "name": name,
            "kind": 1,  # INTERNAL
            "startTimeUnixNano": _nanos(start),
            "endTimeUnixNano": _nanos(end),
            "attributes": attrs,
            "status": {"code": 1},
        }
        if parent:
            s["parentSpanId"] = parent
        return s

    metadata = trace.get("metadata") or {}
    # Langfuse OTel attribute conventions (langfuse.com/integrations/native/opentelemetry):
    # trace-level: langfuse.trace.input / langfuse.trace.metadata.<k> / langfuse.session.id
    # span-level:  langfuse.observation.output / langfuse.observation.metadata.<k>
    root_attrs = []
    if metadata.get("collection"):
        root_attrs.append(_kv("langfuse.session.id", metadata["collection"]))
    if trace.get("input") is not None:
        root_attrs.append(_json_attr("langfuse.trace.input", trace["input"]))
    for key, value in metadata.items():
        if key == "collection":
            continue
        root_attrs.append(_kv(f"langfuse.trace.metadata.{key}", _scalar(value)))

    spans = [span(root_span_id, trace.get("name", "query"),
                  trace["timestamp"],
                  _parse_ts(record.get("finished_at") or record["started_at"]),
                  None, root_attrs)]

    # Ids are positional: several stages share a name (every llm_call does),
    # so names cannot key the id map. Only uniquely-named stages are indexed,
    # and they are what an llm_call hangs off.
    stage_spans = mapped["spans"]
    stage_ids = [_stable_span_id(trace["id"], index) for index in range(len(stage_spans))]
    unique_stage_ids = {
        s["name"].removeprefix("stage:"): sid
        for sid, s in zip(stage_ids, stage_spans)
        if s["name"] != "stage:llm_call"
    }

    call_ordinals: Counter = Counter()
    for stage_id, stage_span in zip(stage_ids, stage_spans):
        recorded_name = stage_span["name"]
        stage_name = recorded_name.removeprefix("stage:")
        data = stage_span.get("metadata") or {}
        display_name = recorded_name

        child_attrs: List[Dict[str, Any]] = [
            _kv("langfuse.observation.type", _observation_type(stage_name))
        ]
        if stage_name == "llm_call":
            # A stage may make a dozen calls; without a distinct name they are
            # indistinguishable in the waterfall.
            purpose = data.get("purpose") or "llm"
            call_ordinals[purpose] += 1
            display_name = f"{purpose}#{call_ordinals[purpose]}"
            if data.get("prompt"):
                child_attrs.append(_json_attr("langfuse.observation.input", data["prompt"]))
            if data.get("response"):
                child_attrs.append(_json_attr("langfuse.observation.output", data["response"]))
            child_attrs.extend(_generation_attrs(data))
        elif stage_span.get("output"):
            child_attrs.append(_json_attr("langfuse.observation.output", stage_span["output"]))
        for key, value in data.items():
            # chunks would bloat the panel; prompt/response are already sent as
            # the observation's input/output.
            if key in ("chunks", "prompt", "response"):
                continue
            child_attrs.append(
                _kv(f"langfuse.observation.metadata.{key}", _scalar(value))
            )
        parent = root_span_id
        if stage_name == "llm_call":
            parent = unique_stage_ids.get(data.get("purpose") or "", root_span_id)
        spans.append(
            span(
                stage_id,
                display_name,
                stage_span["start_time"],
                stage_span["end_time"],
                parent,
                child_attrs,
            )
        )

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _kv("service.name", "modular-rag"),
                    ]
                },
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }


def _credentials() -> tuple:
    """Return (host, public_key, secret_key), raising if the keys are absent."""
    host = (os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set (repo-root .env is loaded automatically)"
        )
    return host, public_key, secret_key


# USD per token (Langfuse's unit for TOKENS). Langfuse ships prices for the big
# hosted providers only; GLM, DeepSeek and local Ollama models are absent, so
# their tokens would otherwise produce no cost. DeepSeek rates match
# scripts/benchmark_rerank.py's table; GLM free tier and local embeddings are 0.
MODEL_PRICES = {
    "glm-4-flash": (0.0, 0.0),
    "glm-4v-flash": (0.0, 0.0),
    "deepseek-chat": (0.28e-06, 0.42e-06),
    # DeepSeek's API echoes "deepseek-flash" as the model name regardless of
    # the requested model, so that is the string Langfuse sees — price it the
    # same as deepseek-chat.
    "deepseek-flash": (0.28e-06, 0.42e-06),
    "deepseek-reasoner": (0.55e-06, 2.19e-06),
    "nomic-embed-text": (0.0, 0.0),
}


def sync_models() -> int:
    """Register this project's models with Langfuse so it can price calls.

    Idempotent: models already present are left alone. Returns the number
    registered.
    """
    import httpx

    host, public_key, secret_key = _credentials()
    auth = (public_key, secret_key)

    existing = set()
    page = 1
    while True:
        listed = httpx.get(
            f"{host}/api/public/models",
            params={"limit": 100, "page": page},
            auth=auth,
            timeout=30.0,
        )
        listed.raise_for_status()
        data = listed.json().get("data", [])
        if not data:
            break
        existing.update(m.get("modelName") for m in data)
        page += 1

    registered = 0
    for name, (input_price, output_price) in MODEL_PRICES.items():
        if name in existing:
            continue
        created = httpx.post(
            f"{host}/api/public/models",
            auth=auth,
            timeout=30.0,
            json={
                "modelName": name,
                "matchPattern": f"(?i)^({re.escape(name)})$",
                "inputPrice": input_price,
                "outputPrice": output_price,
                "unit": "TOKENS",
            },
        )
        created.raise_for_status()
        registered += 1
        print(f"[ok] registered model: {name}")

    print(f"[ok] {registered} registered, {len(existing)} already present at {host}")
    return registered


def _post_otlp(body: Dict[str, Any]) -> bool:
    """POST an OTLP traces body to Langfuse (basic auth pk/sk)."""
    import httpx

    host, public_key, secret_key = _credentials()
    response = httpx.post(
        f"{host}/api/public/otel/v1/traces",
        json=body,
        auth=(public_key, secret_key),
        headers={
            "Content-Type": "application/json",
            # Required for real-time reads on v2 endpoints (else up to 15 min delay)
            "x-langfuse-ingestion-version": "4",
        },
        timeout=30.0,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"OTLP ingestion returned {response.status_code}: {response.text[:300]}"
        )
    return True
