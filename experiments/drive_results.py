"""Read-only Google Drive archive sync for the experiment results viewer."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

_FOLDER_URL_RE = re.compile(r"/folders/([A-Za-z0-9_-]+)")
_RAW_FOLDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class DriveSyncError(RuntimeError):
    """Raised when Drive results cannot be synchronized safely."""


@dataclass(frozen=True)
class RemoteArchive:
    """One ZIP archive found below the configured Drive folder."""

    file_id: str
    name: str
    relative_path: str
    modified_time: str
    size: str
    md5_checksum: str

    @property
    def fingerprint(self) -> str:
        """Return the strongest available stable content fingerprint."""
        return self.md5_checksum or f"{self.modified_time}:{self.size}"


@dataclass(frozen=True)
class SyncReport:
    """Summary of one Drive-to-local synchronization."""

    downloaded: int = 0
    unchanged: int = 0
    removed: int = 0
    ignored: int = 0


def parse_drive_folder_id(folder: str) -> str:
    """Accept a Drive folder URL or raw folder ID and return the ID."""
    value = folder.strip()
    match = _FOLDER_URL_RE.search(value)
    if match:
        return match.group(1)
    if _RAW_FOLDER_ID_RE.fullmatch(value):
        return value
    raise ValueError("expected a Google Drive folder URL or folder ID")


def default_cache_base() -> Path:
    """Return the private per-user cache location."""
    configured = os.environ.get("ROBOCODE_RESULTS_CACHE")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "robocode-results-viewer"


class RcloneResultsSync:
    """Mirror Drive ZIP archives through rclone into an extracted local cache."""

    def __init__(
        self,
        folder: str,
        cache_base: Path,
        remote: str,
        *,
        rclone_binary: str = "rclone",
        config_path: Path | None = None,
    ) -> None:
        self.folder_id = parse_drive_folder_id(folder)
        self.cache_root = cache_base.expanduser().resolve() / self.folder_id
        self.runs_dir = self.cache_root / "runs"
        self.manifest_path = self.cache_root / "manifest.json"
        self.remote = remote.rstrip(":")
        if not self.remote:
            raise ValueError("rclone remote name cannot be empty")
        self.rclone_binary = rclone_binary
        self.config_path = config_path.expanduser() if config_path else None
        self._lock = threading.Lock()

    @staticmethod
    def _path_component(name: str) -> str:
        if name in {"", ".", ".."} or "/" in name or "\\" in name or "\0" in name:
            raise DriveSyncError(f"unsafe Drive file name: {name!r}")
        return name

    def _load_manifest(self) -> dict[str, dict[str, str]]:
        if not self.manifest_path.exists():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return dict(payload.get("archives", {}))
        except (OSError, ValueError, TypeError) as error:
            raise DriveSyncError(
                f"invalid cache manifest: {self.manifest_path}"
            ) from error

    def _write_manifest(self, entries: dict[str, dict[str, str]]) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=".manifest-", suffix=".json", dir=self.cache_root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, "archives": entries}, stream, indent=2)
                stream.write("\n")
            temporary.replace(self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _safe_extract(archive_path: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive_path) as bundle:
            for info in bundle.infolist():
                member = PurePosixPath(info.filename)
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or not member.parts
                    or info.filename.startswith(("/", "\\"))
                    or "\\" in info.filename
                ):
                    raise DriveSyncError(
                        f"unsafe path in {archive_path.name}: {info.filename!r}"
                    )
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise DriveSyncError(
                        f"symbolic link in {archive_path.name}: {info.filename!r}"
                    )
            bundle.extractall(destination)

    def _target_for(self, archive: RemoteArchive) -> Path:
        relative = PurePosixPath(archive.relative_path)
        target = self.runs_dir.joinpath(*relative.parent.parts, relative.stem)
        resolved = target.resolve()
        if not resolved.is_relative_to(self.runs_dir.resolve()):
            raise DriveSyncError(f"unsafe cache target for {archive.relative_path}")
        return resolved

    def _replace_from_archive(self, archive: RemoteArchive, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, download_name = tempfile.mkstemp(
            prefix=f".{target.name}-", suffix=".zip", dir=target.parent
        )
        os.close(fd)
        download = Path(download_name)
        extracted = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        backup = target.with_name(f".{target.name}-previous")
        try:
            self._download(archive, download)
            self._safe_extract(download, extracted)
            if backup.exists():
                shutil.rmtree(backup)
            if target.exists():
                target.replace(backup)
            extracted.replace(target)
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        finally:
            download.unlink(missing_ok=True)
            if extracted.exists():
                shutil.rmtree(extracted)

    def _remove_target(self, relative_target: str) -> None:
        target = (self.cache_root / relative_target).resolve()
        if not target.is_relative_to(self.runs_dir.resolve()):
            raise DriveSyncError(f"unsafe manifest cache target: {relative_target}")
        if target.exists():
            shutil.rmtree(target)

    def sync(self) -> SyncReport:
        """Synchronize remote ZIP archives, preserving unchanged local trees."""
        with self._lock:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            previous = self._load_manifest()
            archives, ignored = self._find_archives()
            targets = [self._target_for(archive) for archive in archives]
            if len(targets) != len(set(targets)):
                raise DriveSyncError(
                    "multiple Drive archives map to the same local cache path"
                )

            current: dict[str, dict[str, str]] = {}
            downloaded = 0
            unchanged = 0
            for archive, target in zip(archives, targets, strict=True):
                relative_target = str(target.relative_to(self.cache_root))
                old = previous.get(archive.file_id)
                if (
                    old
                    and old.get("fingerprint") == archive.fingerprint
                    and old.get("target") == relative_target
                    and target.is_dir()
                ):
                    unchanged += 1
                else:
                    self._replace_from_archive(archive, target)
                    downloaded += 1
                    if old and old.get("target") != relative_target:
                        self._remove_target(str(old.get("target", "")))
                current[archive.file_id] = {
                    **{key: str(value) for key, value in asdict(archive).items()},
                    "fingerprint": archive.fingerprint,
                    "target": relative_target,
                }

            removed = 0
            for file_id, old in previous.items():
                if file_id not in current:
                    self._remove_target(str(old.get("target", "")))
                    removed += 1
            self._write_manifest(current)
            return SyncReport(downloaded, unchanged, removed, ignored)

    def _run_rclone(self, arguments: list[str]) -> str:
        command = [self.rclone_binary, *arguments]
        if self.config_path is not None:
            command.extend(["--config", str(self.config_path)])
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise DriveSyncError(
                f"rclone executable not found: {self.rclone_binary}; "
                "install rclone and run `rclone config`"
            ) from error
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise DriveSyncError(f"rclone failed: {detail}")
        return result.stdout

    def _rclone_source(self, relative_path: str = "") -> str:
        return f"{self.remote}:{relative_path}"

    def _root_folder_arguments(self) -> list[str]:
        return ["--drive-root-folder-id", self.folder_id]

    def _find_archives(self) -> tuple[list[RemoteArchive], int]:
        output = self._run_rclone(
            [
                "lsjson",
                self._rclone_source(),
                "--recursive",
                "--files-only",
                "--hash",
                *self._root_folder_arguments(),
            ]
        )
        try:
            items = json.loads(output)
        except (TypeError, ValueError) as error:
            raise DriveSyncError(
                "rclone returned invalid JSON while listing Drive"
            ) from error
        if not isinstance(items, list):
            raise DriveSyncError("rclone returned an invalid Drive file listing")

        archives: list[RemoteArchive] = []
        ignored = 0
        for item in items:
            if not isinstance(item, dict):
                raise DriveSyncError("rclone returned an invalid Drive file entry")
            relative = PurePosixPath(str(item.get("Path", "")))
            if not relative.parts:
                raise DriveSyncError("rclone returned an empty Drive file path")
            for component in relative.parts:
                self._path_component(component)
            if not relative.name.lower().endswith(".zip"):
                ignored += 1
                continue
            hashes = item.get("Hashes") or {}
            if not isinstance(hashes, dict):
                hashes = {}
            archives.append(
                RemoteArchive(
                    file_id=str(relative),
                    name=relative.name,
                    relative_path=str(relative),
                    modified_time=str(item.get("ModTime", "")),
                    size=str(item.get("Size", "")),
                    md5_checksum=str(hashes.get("MD5") or hashes.get("md5") or ""),
                )
            )
        return sorted(archives, key=lambda archive: archive.relative_path), ignored

    def _download(self, archive: RemoteArchive, destination: Path) -> None:
        self._run_rclone(
            [
                "copyto",
                self._rclone_source(archive.relative_path),
                str(destination),
                *self._root_folder_arguments(),
            ]
        )
