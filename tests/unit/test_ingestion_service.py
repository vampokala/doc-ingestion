from pathlib import Path

from src.web.ingestion_service import save_uploaded_files


class DummyUpload:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


def test_save_uploaded_files_writes_supported_files(tmp_path):
    upload = DummyUpload("sample.txt", b"hello world")
    out = save_uploaded_files(str(tmp_path), [upload])
    assert out[0].status == "queued"
    saved_path = Path(out[0].message)
    assert saved_path.exists()


def test_save_uploaded_files_rejects_unsupported(tmp_path):
    upload = DummyUpload("sample.exe", b"binary")
    out = save_uploaded_files(str(tmp_path), [upload])
    assert out[0].status == "failed"
