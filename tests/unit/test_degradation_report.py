"""Tests for the degradation report's counting logic.

The report is how a silent fallback becomes a number, so the counting rules
(what counts as degraded, what is excluded) are worth pinning down.
"""

from scripts.degradation_report import analyse


def _stage(name, data):
    return {"stage": name, "timestamp": "2026-09-21T10:00:00+00:00", "data": data}


def _trace(stages, trace_type="query"):
    return {"trace_id": "t1", "trace_type": trace_type, "stages": stages}


class TestAnalyse:
    def test_healthy_query_reports_no_degradation(self):
        report = analyse([_trace([_stage("fusion", {"result_count": 5})])])

        assert report["query_traces"] == 1
        assert report["retrieval_fallback"] == 0
        assert report["rerank_fallback"] == 0
        assert report["empty_result_queries"] == 0

    def test_retrieval_fallback_counts_and_names_the_failed_path(self):
        trace = _trace([_stage("retrieval_fallback", {"dense_failed": True, "sparse_failed": False})])

        report = analyse([trace])

        assert report["retrieval_fallback"] == 1
        assert report["retrieval_fallback_rate"] == 1.0
        assert report["failed_paths"] == {"dense": 1}

    def test_rerank_fallback_keeps_the_reason(self):
        trace = _trace([_stage("rerank", {"fallback": True, "fallback_reason": "timeout"})])

        report = analyse([trace])

        assert report["rerank_fallback"] == 1
        assert report["rerank_reasons"] == {"timeout": 1}

    def test_successful_rerank_is_not_a_fallback(self):
        report = analyse([_trace([_stage("rerank", {"method": "jev", "output_count": 5})])])

        assert report["rerank_fallback"] == 0

    def test_query_with_no_fusion_results_is_counted_as_empty(self):
        report = analyse([_trace([_stage("fusion", {"result_count": 0})])])

        assert report["empty_result_queries"] == 1

    def test_ingestion_traces_are_excluded_from_the_rate(self):
        traces = [
            _trace([_stage("retrieval_fallback", {"dense_failed": True})]),
            _trace([_stage("load", {})], trace_type="ingestion"),
        ]

        report = analyse(traces)

        assert report["traces_total"] == 2
        assert report["query_traces"] == 1, "ingestion must not dilute the rate"
        assert report["retrieval_fallback_rate"] == 1.0

    def test_rate_is_zero_when_there_are_no_query_traces(self):
        report = analyse([_trace([], trace_type="ingestion")])

        assert report["query_traces"] == 0
        assert report["retrieval_fallback_rate"] == 0.0

    def test_stage_without_data_is_ignored(self):
        report = analyse([_trace([{"stage": "retrieval_fallback", "timestamp": "x"}])])

        assert report["retrieval_fallback"] == 1
        assert report["failed_paths"] == {}

    def test_empty_input_is_safe(self):
        report = analyse([])

        assert report["traces_total"] == 0
        assert report["retrieval_fallback_rate"] == 0.0
