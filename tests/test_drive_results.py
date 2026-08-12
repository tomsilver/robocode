"""Tests for read-only Google Drive archive synchronization."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from experiments.drive_results import (
    DriveSyncError,
    RcloneResultsSync,
    parse_drive_folder_id,
)


def _archive(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)


def _sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    listing: list[dict[str, object]],
    sources: dict[str, Path],
) -> RcloneResultsSync:
    sync = RcloneResultsSync(
        "root-folder",
        tmp_path / "cache",
        "robocode-drive",
    )

    def _run(arguments: list[str]) -> str:
        if arguments[0] == "lsjson":
            return json.dumps(listing)
        assert arguments[0] == "copyto"
        relative_path = arguments[1].split(":", 1)[1]
        shutil.copyfile(sources[relative_path], arguments[2])
        return ""

    monkeypatch.setattr(sync, "_run_rclone", _run)
    return sync


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
    listing: list[dict[str, object]] = [
        {
            "Path": "unsafe.zip",
            "Name": "unsafe.zip",
            "Size": archive.stat().st_size,
            "ModTime": "2026-08-12T10:00:00Z",
            "Hashes": {"MD5": "unsafe-content"},
        }
    ]
    sync = _sync(tmp_path, monkeypatch, listing, {"unsafe.zip": archive})

    with pytest.raises(DriveSyncError, match="unsafe path"):
        sync.sync()

    assert not (tmp_path / "outside.txt").exists()


def test_sync_recurses_and_preserves_local_gifs_for_unchanged_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged archives retain viewer-generated GIFs in the local cache."""
    source = tmp_path / "experiment.zip"
    _archive(source, {"s42/results.json": "{}", "s42/.hydra/config.yaml": "seed: 42"})
    listing: list[dict[str, object]] = [
        {
            "Path": "Preliminary/experiment-id.zip",
            "Name": "experiment-id.zip",
            "Size": source.stat().st_size,
            "ModTime": "2026-08-12T10:00:00Z",
            "Hashes": {"MD5": "content-v1"},
        },
        {
            "Path": "README",
            "Name": "README",
            "Size": 4,
            "ModTime": "2026-08-12T10:00:00Z",
        },
    ]
    sync = _sync(
        tmp_path,
        monkeypatch,
        listing,
        {"Preliminary/experiment-id.zip": source},
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing a remote archive removes only its corresponding cache tree."""
    source = tmp_path / "experiment.zip"
    _archive(source, {"results.json": "{}"})
    listing: list[dict[str, object]] = [
        {
            "Path": "experiment-id.zip",
            "Name": "experiment-id.zip",
            "Size": source.stat().st_size,
            "ModTime": "2026-08-12T10:00:00Z",
            "Hashes": {"MD5": "content-v1"},
        }
    ]
    sync = _sync(
        tmp_path,
        monkeypatch,
        listing,
        {"experiment-id.zip": source},
    )
    sync.sync()
    extracted = sync.runs_dir / "experiment-id"
    assert extracted.is_dir()

    listing.clear()
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
