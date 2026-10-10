import os
from pathlib import Path

import pytest
import requests


def ollama_available() -> bool:
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="session")
def require_ollama():
    if not ollama_available():
        pytest.skip("Ollama server not reachable at localhost:11434")
