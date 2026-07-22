from pathlib import Path
from types import SimpleNamespace

import pytest

from dungeon_apprentice import artifacts


class _FakeModel:
    def __init__(self, payload: bytes = b"model") -> None:
        self.payload = payload

    def save(self, destination: Path) -> None:
        Path(destination).write_bytes(self.payload)


def test_model_save_and_latest_copy_are_atomic(tmp_path: Path) -> None:
    checkpoint = artifacts.atomic_model_save(_FakeModel(), tmp_path / "model")
    latest = artifacts.atomic_copy_file(checkpoint, tmp_path / "latest.zip")

    assert checkpoint.read_bytes() == b"model"
    assert latest.read_bytes() == b"model"
    assert artifacts.file_sha256(checkpoint) == artifacts.file_sha256(latest)
    assert not list(tmp_path.glob(".*.tmp*"))


def test_disk_space_guard_reports_available_and_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = SimpleNamespace(total=100, used=90, free=10)
    monkeypatch.setattr(artifacts.shutil, "disk_usage", lambda _path: usage)

    assert artifacts.ensure_disk_space(tmp_path, 10) == 10
    with pytest.raises(OSError, match="insufficient disk space"):
        artifacts.ensure_disk_space(tmp_path, 11)


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "nested/run"])
def test_run_name_cannot_escape_run_root(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="beneath run root"):
        artifacts.create_run_directory(tmp_path, name)
