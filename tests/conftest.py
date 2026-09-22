"""Pytest configuration and shared fixtures.

This module contains pytest configuration and fixtures that are shared
across all test modules.
"""

import sys
from pathlib import Path

import pytest

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory path.
    
    Returns:
        Path to the project root directory.
    """
    return PROJECT_ROOT


@pytest.fixture
def sample_documents_dir(project_root: Path) -> Path:
    """Return the sample documents directory path.
    
    Args:
        project_root: The project root directory path.
        
    Returns:
        Path to the sample documents directory.
    """
    return project_root / "tests" / "fixtures" / "sample_documents"


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the config directory path.
    
    Args:
        project_root: The project root directory path.
        
    Returns:
        Path to the config directory.
    """
    return project_root / "config"


@pytest.fixture(autouse=True)
def _no_langfuse_network(monkeypatch):
    """Keep the test suite off the network and out of the real Langfuse project.

    TraceCollector reports every trace it persists, and the credentials for a
    developer's own project live in .env — so without this, merely running the
    tests would post fake traces into a real project. Tests that need to check
    the reporting behaviour patch the sink themselves.

    The collector bound ``send_trace`` into its own namespace at import time,
    so patching the sink module alone would not stop it.
    """
    from src.observability import langfuse_sink

    monkeypatch.setattr("src.core.trace.trace_collector.send_trace", lambda record: False)
    monkeypatch.setattr(langfuse_sink, "send_trace", lambda record: False)
    monkeypatch.setattr(langfuse_sink, "_post_otlp", lambda body: False)
