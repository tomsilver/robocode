"""Prepare trusted Python dependencies before starting isolated agent execution."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def clean_apptainer_env() -> dict[str, str]:
    """Never pass provider secrets or Apptainer override variables to children."""
    return {
        key: os.environ[key]
        for key in (
            "HOME",
            "PATH",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "TERM",
            "TMPDIR",
        )
        if key in os.environ
    }


def prepared_environment(sif: Path, binds: list[str], *, include_bilevel: bool) -> Path:
    """Cache a venv prepared without agent files, credentials, or session mounts.

    The preparation container is intentionally network-capable, but runs only the
    trusted locked installer. Its completed venv is mounted read-only for agents.
    Cache entries are never populated from an agent's writable container overlay.
    """
    from robocode.utils.docker_sandbox import (  # pylint: disable=import-outside-toplevel
        _find_repo_root,
    )

    root = _find_repo_root()
    cache = root / ".apptainer-env-cache"
    cache.mkdir(mode=0o700, exist_ok=True)
    digest = hashlib.sha256()
    digest.update(str(sif.resolve()).encode())
    digest.update(str((sif.stat().st_size, sif.stat().st_mtime_ns)).encode())
    digest.update(str(include_bilevel).encode())
    for path in (
        root / "pyproject.toml",
        root / "uv.lock",
        root / "third-party/kindergarden/pyproject.toml",
    ):
        digest.update(path.read_bytes())
    for bind in binds:
        directory = Path(bind.split(":", 1)[0])
        if directory.is_dir():
            for metadata in sorted(directory.rglob("pyproject.toml")):
                digest.update(str(metadata.relative_to(directory)).encode())
                digest.update(metadata.read_bytes())
    key = digest.hexdigest()
    destination = cache / key
    with (cache / (key + ".lock")).open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "READY").is_file():
            return destination / "venv"
        if destination.exists():
            shutil.rmtree(destination)
        temporary = Path(tempfile.mkdtemp(prefix="prepare-", dir=cache))
        cmd = [
            "apptainer",
            "exec",
            "--userns",
            "--containall",
            "--cleanenv",
            "--no-home",
            "--writable-tmpfs",
            "--pwd",
            "/robocode",
            "--bind",
            f"{temporary}:/prepared",
        ]
        for bind in binds:
            cmd += ["--bind", bind]
        # Seed from the image to avoid redownloading its heavy runtime packages.
        script = (
            "cp -a /robocode/.venv /prepared/venv && "
            "UV_PROJECT_ENVIRONMENT=/prepared/venv "
            "uv sync --frozen --python /usr/bin/python3.11"
        )
        if include_bilevel:
            script += " --extra bilevel"
        cmd += [str(sif.resolve()), "/bin/sh", "-ec", script]
        proc = subprocess.run(
            cmd,
            env=clean_apptainer_env(),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        (temporary / "prepare.log").write_text(
            proc.stdout + proc.stderr, encoding="utf-8"
        )
        if proc.returncode:
            raise RuntimeError(
                "Trusted dependency preparation failed; inspect "
                f"{temporary / 'prepare.log'}"
            )
        # uv created console scripts for /prepared/venv; the runtime bind uses
        # that same path. The agent interpreter remains /robocode/.venv/python
        # via a second read-only bind for existing experiment configuration.
        (temporary / "READY").write_text("1\n", encoding="utf-8")
        temporary.rename(destination)
        return destination / "venv"
