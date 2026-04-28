"""HF Spaces entrypoint.

Bootstraps sample documents on cold-start, starts the FastAPI server in
a background thread (HF Spaces runs a single process), then delegates
to the Streamlit UI.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Ensure repo root is importable regardless of where HF Spaces runs this.
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Set demo mode so the UI shows the banner and disables uploads.
os.environ.setdefault("DOC_PROFILE", "demo")

# Disable auth so the demo doesn't require an API key header.
os.environ.setdefault("DOC_API_KEYS", "demo-key")

# No local Ollama daemon on Spaces — users supply cloud provider keys.
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

# Bootstrap sample documents on first run.
from spaces.bootstrap_demo import bootstrap_if_needed  # noqa: E402

bootstrap_if_needed()


def _start_api() -> None:
    """Run the FastAPI server in a background daemon thread."""
    import uvicorn  # type: ignore[import-untyped]
    from src.api.main import app as fastapi_app  # noqa: E402

    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")


_api_thread = threading.Thread(target=_start_api, daemon=True)
_api_thread.start()

# Give the API a moment to bind before Streamlit renders its first page.
time.sleep(3)

# Hand off to the main Streamlit app.
from src.web.streamlit_app import main  # noqa: E402

main()
