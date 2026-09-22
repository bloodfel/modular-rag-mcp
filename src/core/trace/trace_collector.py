"""Trace collector – receives finished TraceContext and persists them.

The collector is the bridge between in-memory TraceContext objects and the
two places a finished trace goes: the on-disk JSON Lines log (authoritative,
always written) and Langfuse (a remote sink, best-effort). It is
intentionally decoupled from the logging module so that trace persistence
remains predictable and testable.
"""

import json
import logging
from pathlib import Path

from src.core.settings import resolve_path
from src.core.trace.trace_context import TraceContext
from src.observability.langfuse_sink import send_trace

logger = logging.getLogger(__name__)

# Default absolute path for traces file (CWD-independent)
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceCollector:
    """Collects finished traces and appends them to a JSON Lines file.

    Args:
        traces_path: File path for the ``traces.jsonl`` output.
            Parent directories are created automatically.
    """

    def __init__(self, traces_path: str | Path = _DEFAULT_TRACES_PATH) -> None:
        self._path = Path(traces_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def collect(self, trace: TraceContext) -> None:
        """Persist a single trace, then report it to Langfuse.

        The local JSON Lines write comes first and is never skipped: the file
        is the record of truth that Streamlit, the degradation report and the
        evaluation scripts read. Langfuse is a second sink on top of it.

        If the trace has not been finished yet, ``finish()`` is called
        automatically so the output always contains timing data.

        Args:
            trace: A populated :class:`TraceContext`.
        """
        if trace.finished_at is None:
            trace.finish()

        record = trace.to_dict()
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            logger.exception("Failed to write trace %s", trace.trace_id)

        # Reporting is best-effort: collect() runs on the query and ingestion
        # paths, so a broken sink must not break the work it observes.
        # ponytail: synchronous send, roughly 0.3-1s per trace, and a no-op when
        # credentials are absent. Move it to a background thread if query
        # latency ever outweighs having the trace land before the process exits.
        try:
            send_trace(record)
        except Exception:
            logger.exception("Langfuse reporting failed for trace %s", trace.trace_id)

    @property
    def path(self) -> Path:
        """Return the resolved path of the traces file."""
        return self._path
