"""Auto-start Ollama server for local model serving."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import urllib.request

logger = logging.getLogger(__name__)

_OLLAMA_URL = "http://localhost:11434"


def ensure_ollama(keep_alive: str = "5m") -> None:
    """Start an Ollama server if one isn't already running.

    The server inherits ``CUDA_VISIBLE_DEVICES`` from the current
    environment, so set that before calling to restrict GPU usage.
    ``OLLAMA_KEEP_ALIVE`` controls how long models stay loaded in GPU
    memory after the last request.
    """
    # Already reachable — nothing to do.
    try:
        urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=2)  # noqa: S310
        logger.info("Ollama already running at %s", _OLLAMA_URL)
        return
    except OSError:
        pass

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        raise RuntimeError(
            "Ollama binary not found on PATH. "
            "Install it: curl -fsSL https://ollama.com/install.sh | sh"
        )

    env = os.environ.copy()
    env["OLLAMA_KEEP_ALIVE"] = keep_alive

    gpus = env.get("CUDA_VISIBLE_DEVICES", "all")
    logger.info("Starting Ollama server (keep_alive=%s, gpus=%s)", keep_alive, gpus)
    proc = subprocess.Popen(
        [ollama_bin, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for the server to become ready.
    for _ in range(30):
        try:
            urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=2)  # noqa: S310
            logger.info("Ollama server ready (pid=%d)", proc.pid)
            return
        except OSError:
            time.sleep(1)

    raise RuntimeError(f"Ollama server failed to start within 30s (pid={proc.pid})")
