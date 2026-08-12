"""Tests for read-only Google Drive archive synchronization."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from experiments.drive_results import (
    DRIVE_FOLDER_MIME_TYPE,
    DriveResultsSync,
    DriveSyncError,
    parse_drive_folder_id,
)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def execute(self) -> dict[str, Any]:
        return self.payload


class _Files:
    def __init__(self, children: dict[str, list[dict[str, Any]]]) -> None:
        self.children = children

    def list(self, **kwargs: Any) -> _Response:
        parent = kwargs["q"].split("'", 2)[1]
        return _Response({"files": self.children.get(parent, [])})


class _Drive:
    def __init__(self, children: dict[str, list[dict[str, Any]]]) -> None:
        self._files = _Files(children)

    def files(self) -> _Files:
        return self._files


def _archive(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)


def _sync(tmp_path: Path, children: dict[str, list[dict[str, Any]]]) -> DriveResultsSync:
    return DriveResultsSync(
        "root-folder",
        tmp_path / "cache",
        tmp_path / "credentials.json",
        tmp_path / "token.json",
        service=_Drive(children),
    )


def test_parse_drive_folder_url_or_id() -> None:
    assert parse_drive_folder_id("folder_123") == "folder_123"
    assert (
        parse_drive_folder_id(
            "https://drive.google.com/drive/folders/folder_123?usp=sharing"
        )
        == "folder_123"
    )
    with pytest.raises(ValueError, match="folder URL"):
        parse_drive_folder_id("https://example.com/not-drive")


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    _archive(archive, {"../outside.txt": "no"})

    with pytest.raises(DriveSyncError, match="unsafe path"):
        DriveResultsSync._safe_extract(archive, tmp_path / "destination")

    assert not (tmp_path / "outside.txt").exists()


def test_sync_recurses_and_preserves_local_gifs_for_unchanged_archives(
    tmp_path: Path,
) -> None:
    source = tmp_path / "experiment.zip"
    _archive(source, {"s42/results.json": "{}", "s42/.hydra/config.yaml": "seed: 42"})
    children = {
        "root-folder": [
            {
                "id": "preliminary",
                "name": "Preliminary",
                "mimeType": DRIVE_FOLDER_MIME_TYPE,
            },
            {"id": "notes", "name": "README", "mimeType": "text/plain"},
        ],
        "preliminary": [
            {
                "id": "archive-1",
                "name": "experiment-id.zip",
                "mimeType": "application/zip",
                "modifiedTime": "2026-08-12T10:00:00Z",
                "size": str(source.stat().st_size),
                "md5Checksum": "content-v1",
                "capabilities": {"canDownload": True},
            }
        ],
    }
    sync = _sync(tmp_path, children)
    sync._download = lambda _archive_info, destination: shutil.copyfile(  # type: ignore[method-assign]
        source, destination
    )

    first = sync.sync()
    extracted = sync.runs_dir / "Preliminary" / "experiment-id"
    assert first.downloaded == 1
    assert first.ignored == 1
    assert (extracted / "s42" / "results.json").is_file()

    local_gif = extracted / "s42" / "videos" / "episode_0.gif"
    local_gif.parent.mkdir()
    local_gif.write_bytes(b"GIF89a")
    second = sync.sync()

    assert second.downloaded == 0
    assert second.unchanged == 1
    assert local_gif.read_bytes() == b"GIF89a"


def test_sync_removes_cache_for_remote_archive_deleted_from_drive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "experiment.zip"
    _archive(source, {"results.json": "{}"})
    children = {
        "root-folder": [
            {
                "id": "archive-1",
                "name": "experiment-id.zip",
                "mimeType": "application/zip",
                "modifiedTime": "2026-08-12T10:00:00Z",
                "size": str(source.stat().st_size),
                "md5Checksum": "content-v1",
                "capabilities": {"canDownload": True},
            }
        ]
    }
    sync = _sync(tmp_path, children)
    sync._download = lambda _archive_info, destination: shutil.copyfile(  # type: ignore[method-assign]
        source, destination
    )
    sync.sync()
    extracted = sync.runs_dir / "experiment-id"
    assert extracted.is_dir()

    children["root-folder"].clear()
    report = sync.sync()

    assert report.removed == 1
    assert not extracted.exists()
