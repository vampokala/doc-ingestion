"""Helpers for Streamlit upload and ingest workflow."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from src.ingest import collect_files, ingest

_LOCK = threading.Lock()
_SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".html"}


@dataclass
class IngestFileResult:
    filename: str
    status: str
    message: str = ""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_uploaded_files(upload_dir: str, files: Iterable[object]) -> List[IngestFileResult]:
    os.makedirs(upload_dir, exist_ok=True)
    results: List[IngestFileResult] = []
    for f in files:
        name = getattr(f, "name", "unknown")
        data = getattr(f, "getvalue", lambda: b"")()
        ext = Path(name).suffix.lower()
        if ext not in _SUPPORTED_EXTS:
            results.append(IngestFileResult(filename=name, status="failed", message=f"Unsupported type: {ext}"))
            continue
        digest = _sha256_bytes(data)[:12]
        out_path = os.path.join(upload_dir, f"{digest}_{Path(name).name}")
        if os.path.exists(out_path):
            results.append(IngestFileResult(filename=name, status="skipped", message="Duplicate upload"))
            continue
        with open(out_path, "wb") as w:
            w.write(data)
        results.append(IngestFileResult(filename=name, status="queued", message=out_path))
    return results


def run_ingest(upload_dir: str) -> dict:
    with _LOCK:
        files = collect_files(upload_dir)
        if not files:
            return {"processed_files": 0, "status": "empty"}
        ingest(upload_dir)
        return {"processed_files": len(files), "status": "ok"}
