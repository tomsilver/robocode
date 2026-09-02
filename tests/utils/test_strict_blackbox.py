"""Tests for the strict blackbox variant: config, prompt, server, and imports."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.spaces import Box

from robocode.approaches.agentic_approach import AgenticApproach
from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.docker_sandbox import _docker_run_prefix
from robocode.utils.env_client import BlackboxEnv, SpaceInfo, _BlackboxObservationSpace
from robocode.utils.env_server_runtime import _dispatch, _HandleRegistry
from robocode.utils.episode import load_generated_approach
from robocode.utils.strict_blackbox import StrictImportError, check_strict_imports

_ENV_CFG = '{"_target_": "unused.ForValidation"}'


class _FakeSocket:
    """Socket stub sufficient for constructing a client without a server."""

    def __init__(self) -> None:
        self.file = io.BytesIO()

    def makefile(self, _mode: str) -> io.BytesIO:
        """Return an inert file-like transport."""
        return self.file

    def close(self) -> None:
        """Close the inert transport."""
        self.file.close()


class _DummyEnv:
    """Tiny env surface for testing strict command dispatch without a server."""

    def reset(
        self, seed: int | None, options: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a deterministic observation for *seed*."""
        del options
        return np.array([seed or 0], dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Echo *action* as the next observation."""
        return np.asarray(action, dtype=np.float32), 1.0, False, False, {}

    def get_state(self) -> np.ndarray:
        """Return a value used to prove this method is blocked."""
        return np.zeros(1, dtype=np.float32)


def _strict_args(**overrides: Any) -> dict[str, Any]:
    space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    args: dict[str, Any] = {
        "action_space": space,
        "observation_space": space,
        "seed": 0,
        "primitives": {},
        "backend": DEFAULT_BACKEND_CFG,
        "container_backend": "docker",
        "blackbox": True,
        "blackbox_strict": True,
        "env_cfg": _ENV_CFG,
        "mcp_tools": (),
    }
    args.update(overrides)
    return args


def test_strict_blackbox_prompt_describes_only_the_generic_surface() -> None:
    """Strict mode advertises reset/step, the allowlist, and the strict python."""
    approach = AgenticApproach(**_strict_args())
    prompt, _, _ = approach._build_agentic_prompts()  # pylint: disable=protected-access
    assert "STRICT BLACK BOX" in prompt
    assert "Only `reset` and `step` are available" in prompt
    assert "NumPy, SciPy" in prompt
    assert "checked against that list" in prompt
    assert "/opt/robocode-strict/bin/python" in prompt
    assert "set_state" not in prompt
    assert "make_primitives" not in prompt


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"blackbox": False}, "requires blackbox=true"),
        ({"container_backend": "local"}, "requires container_backend=docker"),
        ({"primitives": {"helper": object()}}, "exposes no primitives"),
        ({"mcp_tools": ("render_state",)}, "exposes no MCP"),
    ],
)
def test_strict_blackbox_rejects_capability_leaks(
    overrides: dict[str, Any], message: str
) -> None:
    """Strict mode fails fast when a conflicting capability is configured."""
    with pytest.raises(ValueError, match=message):
        AgenticApproach(**_strict_args(**overrides))


def test_strict_server_allows_only_reset_and_step(tmp_path: Path) -> None:
    """The host rejects helper commands independently of client behavior."""
    env = _DummyEnv()
    registry = _HandleRegistry()
    reset = _dispatch(env, {"cmd": "reset", "seed": 4}, tmp_path, registry, strict=True)
    assert reset["obs"]["__ndarray__"] == [4.0]
    stepped = _dispatch(
        env,
        {"cmd": "step", "action": {"__ndarray__": [0.5], "dtype": "float32"}},
        tmp_path,
        registry,
        strict=True,
    )
    assert stepped["reward"] == 1.0
    with pytest.raises(ValueError, match="only reset and step"):
        _dispatch(env, {"cmd": "get_state"}, tmp_path, registry, strict=True)


def test_strict_client_uses_plain_space_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict clients do not attach host-backed devectorization helpers."""
    monkeypatch.setattr("socket.create_connection", lambda _address: _FakeSocket())
    box = {
        "type": "Box",
        "shape": [1],
        "low": [-1.0],
        "high": [1.0],
        "dtype": "float32",
    }
    client = BlackboxEnv(
        {
            "strict": True,
            "observation_space": box,
            "action_space": box,
            "max_steps": 10,
            "primitives": [],
            "token": "unused",
            "host": "unused",
            "port": 0,
        }
    )
    assert isinstance(client.observation_space, SpaceInfo)
    assert not isinstance(client.observation_space, _BlackboxObservationSpace)
    assert not hasattr(client.observation_space, "devectorize")
    client._file.close()  # pylint: disable=protected-access
    client._sock.close()  # pylint: disable=protected-access


def test_strict_docker_launch_has_no_project_mounts(tmp_path: Path) -> None:
    """A strict launch mounts only the sandbox and names the env-server port."""
    command = _docker_run_prefix(
        "strict-test",
        "strict-image",
        tmp_path,
        None,
        None,
        None,
        [],
        [],
        map_host_gateway=True,
        env_server_port=43210,
    )
    joined = " ".join(command)
    assert f"{tmp_path.resolve()}:/sandbox" in joined
    assert "ROBOCODE_FIREWALL_HOST_PORT=43210" in command
    assert "host.docker.internal:host-gateway" in command
    assert "/robocode/src" not in joined
    assert "kindergarden" not in joined
    assert "ss-pybullet" not in joined


def _write(root: Path, name: str, source: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_strict_imports_accept_stdlib_allowed_packages_and_siblings(
    tmp_path: Path,
) -> None:
    """Stdlib, numpy/scipy, plain and relative sibling imports all pass."""
    _write(tmp_path, "helper.py", "import math\nfrom scipy import optimize\nK = 1\n")
    _write(tmp_path, "pkg/__init__.py", "from . import inner\n")
    _write(tmp_path, "pkg/inner.py", "import json\n")
    entry = _write(
        tmp_path,
        "approach.py",
        "import numpy as np\nimport helper\nfrom pkg import inner\n"
        "class GeneratedApproach:\n    pass\n",
    )
    assert check_strict_imports(entry) == {"numpy", "scipy"}


@pytest.mark.parametrize(
    ("files", "offender"),
    [
        ({"approach.py": "import shapely\n"}, "approach.py:1: import shapely"),
        (
            {
                "approach.py": "def f():\n    try:\n        import pybullet\n"
                "    except ImportError:\n        pass\n"
            },
            "approach.py:3: import pybullet",
        ),
        (
            {"approach.py": "import helper\n", "helper.py": "from kinder import x\n"},
            "helper.py:1: import kinder",
        ),
        (
            {"approach.py": "import helper, robocode\n", "helper.py": "K = 1\n"},
            "approach.py:1: import robocode",
        ),
        ({"approach.py": "import importlib\n"}, "import importlib"),
        ({"approach.py": "m = __import__('os')\n"}, "approach.py:1: __import__"),
        ({"approach.py": "from env_client import make_env\n"}, "import env_client"),
    ],
)
def test_strict_imports_reject_everything_else(
    tmp_path: Path, files: dict[str, str], offender: str
) -> None:
    """Nested, sibling-transitive, dynamic, and client imports are all named."""
    for name, source in files.items():
        _write(tmp_path, name, source)
    with pytest.raises(StrictImportError, match=offender):
        check_strict_imports(tmp_path / "approach.py")


def test_loader_runs_the_strict_check_before_exec(tmp_path: Path) -> None:
    """A strict program with a host-only import never gets executed."""
    entry = _write(
        tmp_path,
        "approach.py",
        "import robocode\nraise SystemExit('must not run')\n",
    )
    space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    with pytest.raises(StrictImportError):
        load_generated_approach(entry, space, space, {}, strict_imports=True)


def test_strict_loader_rejects_sibling_that_shadows_cached_host_module(
    tmp_path: Path,
) -> None:
    """A sibling cannot disguise a host module already present in sys.modules."""
    _write(tmp_path, "robocode.py", "LOCAL = True\n")
    entry = _write(
        tmp_path,
        "approach.py",
        "import robocode\n"
        "class GeneratedApproach:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        assert robocode.LOCAL\n",
    )
    space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    with pytest.raises(StrictImportError, match="cached host module robocode"):
        load_generated_approach(entry, space, space, {}, strict_imports=True)
