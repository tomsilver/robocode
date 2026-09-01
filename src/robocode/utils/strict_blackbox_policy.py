"""Host proxy for a policy evaluated in the generic strict-blackbox image."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import IO, Any

from robocode.utils.env_server import decode, encode

STRICT_BLACKBOX_IMAGE = "robocode-strict-blackbox"
STRICT_BLACKBOX_PYTHON = "/opt/robocode-strict/bin/python"
STRICT_BLACKBOX_WORKER = "/opt/strict_blackbox_runtime.py"


class StrictBlackboxPolicy:
    """Run ``GeneratedApproach`` behind a networkless Docker RPC boundary."""

    def __init__(
        self,
        policy_path: Path,
        *,
        action_space: dict[str, Any],
        observation_space: dict[str, Any],
        image: str = STRICT_BLACKBOX_IMAGE,
    ) -> None:
        self._policy_dir = policy_path.resolve().parent
        self._action_space = action_space
        self._observation_space = observation_space
        self._image = image
        self._stderr: IO[str] | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._closed = True
        self._start()

    def _start(self) -> None:
        """Start a fresh policy process (one is reused until failure/timeout)."""
        self._stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(  # pylint: disable=consider-using-with
                [
                    "docker",
                    "run",
                    "--rm",
                    "-i",
                    "--network=none",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--pids-limit=128",
                    "--user",
                    "node",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,size=64m,mode=1777",
                    "-v",
                    f"{self._policy_dir}:/policy:ro",
                    "-w",
                    "/policy",
                    "--entrypoint",
                    STRICT_BLACKBOX_PYTHON,
                    self._image,
                    STRICT_BLACKBOX_WORKER,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                bufsize=1,
            )
        except BaseException:
            self._stderr.close()
            self._stderr = None
            raise
        self._closed = False
        try:
            self._rpc(
                "init",
                action_space=self._action_space,
                observation_space=self._observation_space,
            )
        except BaseException:
            self.close()
            raise

    def _rpc(self, command: str, **payload: Any) -> Any:
        if (
            self._closed
            or self._proc is None
            or self._proc.stdin is None
            or self._proc.stdout is None
        ):
            raise RuntimeError("Strict blackbox policy container is not running")
        try:
            self._proc.stdin.write(json.dumps({"cmd": command, **payload}) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except BaseException:
            self.close()
            raise
        if not line:
            code = self._proc.poll()
            assert self._stderr is not None
            self._stderr.seek(0)
            details = self._stderr.read().strip()
            self.close()
            raise RuntimeError(
                f"Strict blackbox policy container exited with code {code}"
                + (f":\n{details}" if details else "")
            )
        response = json.loads(line)
        if not response.get("ok"):
            error = RuntimeError(
                "Strict blackbox policy failed inside the generic container:\n"
                + response.get("traceback", response.get("error", "unknown error"))
            )
            self.close()
            raise error
        return decode(response.get("result"))

    def reset(self, state: Any, info: dict[str, Any]) -> None:
        """Start an episode, restarting the container after a prior stop."""
        if self._closed:
            self._start()
        self._rpc("reset", state=encode(state), info=encode(info))

    def get_action(self, state: Any) -> Any:
        """Return the generated policy's action for *state*."""
        return self._rpc("get_action", state=encode(state))

    def update(
        self, state: Any, reward: float, done: bool, info: dict[str, Any]
    ) -> None:
        """Forward transition feedback and stop the worker at episode end."""
        self._rpc(
            "update",
            state=encode(state),
            reward=float(reward),
            done=bool(done),
            info=encode(info),
        )
        if done:
            self.close()

    def close(self) -> None:
        """Stop the attached policy container and close its transport."""
        if self._closed:
            return
        self._closed = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                    self._proc.stdin.flush()
                self._proc.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
        if self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.close()
        if self._proc is not None and self._proc.stdout is not None:
            self._proc.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
        self._proc = None
        self._stderr = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pylint: disable=broad-exception-caught
            # Interpreter shutdown or a partially started Docker client.
            pass
