"""Tests for read-only Google Drive archive synchronization."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from experiments.drive_results import (
    DRIVE_FOLDER_MIME_TYPE,
    DriveResultsSync,
    DriveSyncError,
    RcloneResultsSync,
    parse_drive_folder_id,
)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def execute(self) -> dict[str, Any]:
        """Return the configured fake API payload."""
        return self.payload


class _Files:
    def __init__(self, children: dict[str, list[dict[str, Any]]]) -> None:
        self.children = children

    def list(self, **kwargs: Any) -> _Response:
        """Return the fake children of the requested parent folder."""
        parent = kwargs["q"].split("'", 2)[1]
        return _Response({"files": self.children.get(parent, [])})


class _Drive:
    def __init__(self, children: dict[str, list[dict[str, Any]]]) -> None:
        self._files = _Files(children)

    def files(self) -> _Files:
        """Return the fake Drive files resource."""
        return self._files


def _archive(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)


def _sync(
    tmp_path: Path, children: dict[str, list[dict[str, Any]]]
) -> DriveResultsSync:
    return DriveResultsSync(
        "root-folder",
        tmp_path / "cache",
        tmp_path / "credentials.json",
        tmp_path / "token.json",
        service=_Drive(children),
    )


def test_parse_drive_folder_url_or_id() -> None:
    """Drive folder URLs and bare IDs resolve to the same folder ID."""
    assert parse_drive_folder_id("folder_123") == "folder_123"
    assert (
        parse_drive_folder_id(
            "https://drive.google.com/drive/folders/folder_123?usp=sharing"
        )
        == "folder_123"
    )
    with pytest.raises(ValueError, match="folder URL"):
        parse_drive_folder_id("https://example.com/not-drive")


def test_safe_extract_rejects_parent_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synchronizing an archive cannot write outside its cache target."""
    archive = tmp_path / "unsafe.zip"
    _archive(archive, {"../outside.txt": "no"})
    children: dict[str, list[dict[str, Any]]] = {
        "root-folder": [
            {
                "id": "unsafe-archive",
                "name": "unsafe.zip",
                "mimeType": "application/zip",
                "modifiedTime": "2026-08-12T10:00:00Z",
                "size": str(archive.stat().st_size),
                "md5Checksum": "unsafe-content",
            }
        ]
    }
    sync = _sync(tmp_path, children)

    def _download(_archive_info: Any, destination: Path) -> None:
        shutil.copyfile(archive, destination)

    monkeypatch.setattr(sync, "_download", _download)

    with pytest.raises(DriveSyncError, match="unsafe path"):
        sync.sync()

    assert not (tmp_path / "outside.txt").exists()


def test_sync_recurses_and_preserves_local_gifs_for_unchanged_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged archives retain viewer-generated GIFs in the local cache."""
    source = tmp_path / "experiment.zip"
    _archive(source, {"s42/results.json": "{}", "s42/.hydra/config.yaml": "seed: 42"})
    children: dict[str, list[dict[str, Any]]] = {
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

    def _download(_archive_info: Any, destination: Path) -> None:
        shutil.copyfile(source, destination)

    monkeypatch.setattr(sync, "_download", _download)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing a remote archive removes only its corresponding cache tree."""
    source = tmp_path / "experiment.zip"
    _archive(source, {"results.json": "{}"})
    children: dict[str, list[dict[str, Any]]] = {
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

    def _download(_archive_info: Any, destination: Path) -> None:
        shutil.copyfile(source, destination)

    monkeypatch.setattr(sync, "_download", _download)
    sync.sync()
    extracted = sync.runs_dir / "experiment-id"
    assert extracted.is_dir()

    children["root-folder"].clear()
    report = sync.sync()

    assert report.removed == 1
    assert not extracted.exists()


def test_rclone_sync_discovers_multiple_archives_and_preserves_local_gifs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rclone backend mirrors nested ZIPs and preserves unchanged GIFs."""
    first_source = tmp_path / "first.zip"
    second_source = tmp_path / "second.zip"
    _archive(first_source, {"s42/results.json": "{}"})
    _archive(second_source, {"s24/results.json": "{}"})
    sources = {
        "first-experiment.zip": first_source,
        "nested/second-experiment.zip": second_source,
    }
    listing = [
        {
            "Path": relative,
            "Name": Path(relative).name,
            "Size": source.stat().st_size,
            "ModTime": "2026-08-12T10:00:00Z",
            "Hashes": {"MD5": f"hash-{index}"},
        }
        for index, (relative, source) in enumerate(sources.items())
    ]
    listing.append(
        {
            "Path": "notes.txt",
            "Name": "notes.txt",
            "Size": 4,
            "ModTime": "2026-08-12T10:00:00Z",
        }
    )
    sync = RcloneResultsSync(
        "folder-id",
        tmp_path / "cache",
        "robocode-drive",
    )
    commands: list[list[str]] = []

    def _run(arguments: list[str]) -> str:
        commands.append(arguments)
        if arguments[0] == "lsjson":
            return json.dumps(listing)
        assert arguments[0] == "copyto"
        relative = arguments[1].split(":", 1)[1]
        shutil.copyfile(sources[relative], arguments[2])
        return ""

    monkeypatch.setattr(sync, "_run_rclone", _run)

    first = sync.sync()
    first_target = sync.runs_dir / "first-experiment"
    second_target = sync.runs_dir / "nested" / "second-experiment"
    assert first.downloaded == 2
    assert first.ignored == 1
    assert (first_target / "s42" / "results.json").is_file()
    assert (second_target / "s24" / "results.json").is_file()
    assert all("--drive-root-folder-id" in command for command in commands)

    local_gif = first_target / "s42" / "videos" / "episode_0.gif"
    local_gif.parent.mkdir()
    local_gif.write_bytes(b"GIF89a")
    commands.clear()
    second = sync.sync()

    assert second.downloaded == 0
    assert second.unchanged == 2
    assert local_gif.read_bytes() == b"GIF89a"
    assert [command[0] for command in commands] == ["lsjson"]
