#!/usr/bin/env python
"""CLI for the Langfuse sink: backfill and model registration.

Reporting is automatic — ``TraceCollector`` sends every trace as it is
written, so nothing here is needed for normal use. This exists for the two
things that are not automatic:

  * backfilling traces that were written while Langfuse was unreachable
  * registering this project's models so Langfuse can price their calls

Usage:
    .venv/bin/python scripts/export_langfuse.py --last 5
    .venv/bin/python scripts/export_langfuse.py --last 1 --type ingestion
    .venv/bin/python scripts/export_langfuse.py --sync-models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability.langfuse_sink import export, sync_models  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill traces.jsonl into Langfuse.")
    parser.add_argument("--last", type=int, default=10, help="Export the last N traces")
    parser.add_argument("--type", choices=["query", "ingestion"], default=None)
    parser.add_argument(
        "--sync-models",
        action="store_true",
        help="Register this project's models (GLM / DeepSeek / Ollama) so Langfuse can price them, then exit",
    )
    args = parser.parse_args()
    if args.sync_models:
        sync_models()
        return 0
    return export(last=args.last, trace_type=args.type)


if __name__ == "__main__":
    sys.exit(main())
