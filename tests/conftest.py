"""Shared fixtures: an isolated database and an offline demo-mode app client."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def isolated_db():
    """Point the app at a throwaway SQLite file before it is imported."""
    tmp = tempfile.mkdtemp()
    os.environ["YOGERA_DB"] = str(Path(tmp) / "test.db")
    os.environ["LLM_PROVIDER"] = "mock"          # deterministic, offline
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LLM_API_KEY"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(ROOT))
    yield


@pytest.fixture(scope="session")
def client(isolated_db):
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415

    return TestClient(app)
