"""HF Spaces entrypoint.

Bootstraps sample documents on cold-start, then delegates to the
main Streamlit app at src/web/streamlit_app.py.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Ensure repo root is importable regardless of where HF Spaces runs this.
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Set demo mode so the UI shows the banner and disables uploads.
os.environ.setdefault("DOC_PROFILE", "demo")

# Disable Ollama provider in demo (no local daemon available on Spaces).
# Users can still override provider via the sidebar session keys.
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

# Bootstrap sample documents on first run.
from spaces.bootstrap_demo import bootstrap_if_needed  # noqa: E402

bootstrap_if_needed()

# Hand off to the main Streamlit app.
from src.web.streamlit_app import main  # noqa: E402

main()
