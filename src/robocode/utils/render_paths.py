"""Shared helpers for building render output filenames.

Both render entry points write PNGs whose names embed an agent-supplied label:
the host-side black-box runtime (:mod:`robocode.utils.env_server_runtime`) and
the in-sandbox local render server (:mod:`robocode.mcp.local_render`). They live
in separate processes (one on the host, one in the container), so the shared
logic lives here rather than being imported from either.
"""

from __future__ import annotations

import re
from pathlib import Path


def safe_label(label: str) -> str:
    """Reduce an agent-supplied label to a filename-safe token.

    The label is interpolated into the render output filename, so a label with
    path separators or ``..`` could otherwise steer the PNG write outside the
    renders directory (and, in the local backend, outside the sandbox on the
    host). Keep only alphanumerics, dashes, and underscores; collapse everything
    else to ``_``.
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label)


def unique_path(directory: Path, stem: str, ext: str = ".png") -> Path:
    """Return ``directory/stem.ext``, appending _1, _2, ...

    if taken.
    """
    candidate = directory / f"{stem}{ext}"
    i = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{i}{ext}"
        i += 1
    return candidate
