"""Tests for what the ingestion trace says when nothing ran.

A skipped file and a crashed pipeline both used to leave the trace empty, so
the dashboard could not tell them apart and reported a healthy skip as "the
document may be corrupted or unsupported". Both cases must now be visible.
"""

from src.core.trace.trace_context import TraceContext
from src.ingestion.pipeline import IngestionPipeline
from tests.unit.test_pipeline_progress import _make_fake_pipeline


def _stage(trace: TraceContext, name: str):
    return next((s for s in trace.stages if s["stage"] == name), None)


class TestSkippedFileIsVisible:
    def test_skip_records_an_integrity_stage(self) -> None:
        fp = _make_fake_pipeline()
        fp.integrity_checker.should_skip.return_value = True
        trace = TraceContext(trace_type="ingestion")

        result = IngestionPipeline.run(fp, "test.pdf", trace=trace)

        assert result.success is True, "a skip is not a failure"
        stage = _stage(trace, "integrity")
        assert stage is not None, "a skip must leave a trace, not an empty one"
        assert stage["data"]["skipped"] is True
        assert stage["data"]["reason"] == "already_processed"
        assert stage["data"]["file_hash"] == "hash123"

    def test_skip_does_not_run_the_pipeline(self) -> None:
        fp = _make_fake_pipeline()
        fp.integrity_checker.should_skip.return_value = True
        trace = TraceContext(trace_type="ingestion")

        IngestionPipeline.run(fp, "test.pdf", trace=trace)

        fp.loader.load.assert_not_called()
        assert [s["stage"] for s in trace.stages] == ["integrity"]

    def test_force_bypasses_the_skip(self) -> None:
        fp = _make_fake_pipeline()
        fp.force = True
        fp.integrity_checker.should_skip.return_value = True
        trace = TraceContext(trace_type="ingestion")

        IngestionPipeline.run(fp, "test.pdf", trace=trace)

        assert fp.integrity_checker.should_skip.called is False
        assert _stage(trace, "load") is not None


class TestFailureIsVisible:
    def test_failure_records_the_error_on_the_trace(self) -> None:
        fp = _make_fake_pipeline()
        fp.loader.load.side_effect = RuntimeError("Unsupported PDF structure")
        trace = TraceContext(trace_type="ingestion")

        result = IngestionPipeline.run(fp, "test.pdf", trace=trace)

        assert result.success is False
        stage = _stage(trace, "pipeline_error")
        assert stage is not None, "the reason must be readable from the trace"
        assert stage["data"]["error"] == "Unsupported PDF structure"
        assert stage["data"]["error_type"] == "RuntimeError"

    def test_failure_names_the_last_completed_stage(self) -> None:
        """So a reader knows how far the run got before dying."""
        fp = _make_fake_pipeline()
        fp.chunker.split_document.side_effect = ValueError("split exploded")
        trace = TraceContext(trace_type="ingestion")

        IngestionPipeline.run(fp, "test.pdf", trace=trace)

        assert _stage(trace, "pipeline_error")["data"]["last_stage_before_error"] == "load"

    def test_run_without_trace_still_works(self) -> None:
        fp = _make_fake_pipeline()
        fp.loader.load.side_effect = RuntimeError("boom")

        result = IngestionPipeline.run(fp, "test.pdf")

        assert result.success is False
