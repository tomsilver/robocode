"""Tests for Ollama server auto-start."""

import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from robocode.utils.backends.ollama_server import ensure_ollama


class TestEnsureOllama:
    """Tests for ensure_ollama()."""

    def test_noop_when_already_running(self) -> None:
        """If Ollama is reachable, do nothing."""
        with patch.object(urllib.request, "urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            ensure_ollama()
        # urlopen called once to check, no Popen spawned.
        mock_urlopen.assert_called_once()

    def test_starts_server_when_not_running(self) -> None:
        """If Ollama is not reachable, start it and wait."""
        call_count = 0

        def urlopen_side_effect(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: not running.
                raise OSError("Connection refused")
            # Subsequent calls: now running.
            return MagicMock()

        with (
            patch.object(urllib.request, "urlopen", side_effect=urlopen_side_effect),
            patch("robocode.utils.backends.ollama_server.shutil.which") as mock_which,
            patch(
                "robocode.utils.backends.ollama_server.subprocess.Popen"
            ) as mock_popen,
            patch("robocode.utils.backends.ollama_server.time.sleep"),
        ):
            mock_which.return_value = "/usr/bin/ollama"
            mock_popen.return_value = MagicMock(pid=12345)
            ensure_ollama(keep_alive="10m")

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == ["/usr/bin/ollama", "serve"]
        assert call_args[1]["env"]["OLLAMA_KEEP_ALIVE"] == "10m"
        assert call_args[1]["start_new_session"] is True

    def test_inherits_cuda_visible_devices(self) -> None:
        """CUDA_VISIBLE_DEVICES from env is passed to the server."""
        call_count = 0

        def urlopen_side_effect(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Connection refused")
            return MagicMock()

        with (
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,3"}),
            patch.object(urllib.request, "urlopen", side_effect=urlopen_side_effect),
            patch("robocode.utils.backends.ollama_server.shutil.which") as mock_which,
            patch(
                "robocode.utils.backends.ollama_server.subprocess.Popen"
            ) as mock_popen,
            patch("robocode.utils.backends.ollama_server.time.sleep"),
        ):
            mock_which.return_value = "/usr/bin/ollama"
            mock_popen.return_value = MagicMock(pid=12345)
            ensure_ollama()

        env = mock_popen.call_args[1]["env"]
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"

    def test_raises_when_ollama_not_found(self) -> None:
        """RuntimeError if ollama binary is not on PATH."""
        with (
            patch.object(urllib.request, "urlopen", side_effect=OSError("refused")),
            patch("robocode.utils.backends.ollama_server.shutil.which") as mock_which,
        ):
            mock_which.return_value = None
            with pytest.raises(RuntimeError, match="not found on PATH"):
                ensure_ollama()

    def test_raises_on_timeout(self) -> None:
        """RuntimeError if server doesn't become ready within timeout."""
        with (
            patch.object(urllib.request, "urlopen", side_effect=OSError("refused")),
            patch("robocode.utils.backends.ollama_server.shutil.which") as mock_which,
            patch(
                "robocode.utils.backends.ollama_server.subprocess.Popen"
            ) as mock_popen,
            patch("robocode.utils.backends.ollama_server.time.sleep"),
        ):
            mock_which.return_value = "/usr/bin/ollama"
            mock_popen.return_value = MagicMock(pid=99999)
            with pytest.raises(RuntimeError, match="failed to start within 30s"):
                ensure_ollama()
