"""Tests for the strict blackbox variant: config, prompt, server, and imports."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.spaces import Box

from robocode import prompts
from robocode.approaches.agentic_approach import AgenticApproach
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.docker_sandbox import _docker_run_prefix, _mcp_prestart_wrapper
from robocode.utils.env_client import BlackboxEnv, SpaceInfo, _BlackboxObservationSpace
from robocode.utils.env_server import env_server_running, write_env_spaces
from robocode.utils.env_server_runtime import _dispatch, _HandleRegistry
from robocode.utils.episode import load_generated_approach
from robocode.utils.strict_blackbox import (
    STRICT_ALLOWED_PACKAGES,
    STRICT_BLACKBOX_MCP_PYTHON,
    STRICT_BLACKBOX_PYTHON,
    StrictImportError,
    check_strict_imports,
    strict_allowlist_description,
)

_ENV_CFG = '{"_target_": "unused.ForValidation"}'
_RENDER_ENV_CFG = {
    "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
    "env_id": "kinder/Motion2D-p0-v0",
}
_STRICT_IMAGE = "robocode-strict-blackbox"


def _strict_image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", _STRICT_IMAGE],
        capture_output=True,
        timeout=10,
        check=False,
    )
    return result.returncode == 0


requires_strict_docker = pytest.mark.skipif(
    not _strict_image_available(),
    reason=f"Docker image {_STRICT_IMAGE!r} is unavailable",
)


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
    assert "No environment, simulator, robotics, or geometry libraries" in prompt
    assert "`env.make_primitives()` returns an empty dict" in prompt
    assert f"may import only {strict_allowlist_description()}" in prompt
    assert "/opt/robocode-strict/bin/python" in prompt
    assert "env.get_state(" not in prompt
    assert "env.set_state(" not in prompt
    assert "devectorize(obs)" not in prompt
    assert "primitives = env.make_primitives()" not in prompt


def test_strict_spec_is_the_blackbox_spec_minus_withheld_helpers() -> None:
    """Strict and legacy blackbox share one spec; only the withheld surface differs."""
    for object_centric in (False, True):
        legacy = prompts.blackbox_interaction_spec(object_centric=object_centric)
        strict = prompts.blackbox_interaction_spec(
            object_centric=object_centric, strict=True
        )
        for shared in (
            "The environment is a BLACK BOX.",
            "obs, reward, terminated, truncated, info = env.step(action)",
            "Parallel test scripts are fine",
            "CRITICAL: `approach.py` itself must NOT import `env_client`",
        ):
            assert shared in legacy
            assert shared in strict
        assert "env.get_state()" in legacy
        assert "env.get_state()" not in strict


def test_strict_allowlist_description_names_every_allowed_package() -> None:
    """The prose the prompt and the rejection share tracks the allowlist constant."""
    description = strict_allowlist_description()
    for package in STRICT_ALLOWED_PACKAGES:
        assert f"`{package}`" in description
    assert "`sys.modules`" in description


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"blackbox": False}, "requires blackbox=true"),
        ({"container_backend": "local"}, "requires container_backend=docker"),
        ({"primitives": {"helper": object()}}, "exposes no primitives"),
    ],
)
def test_strict_blackbox_rejects_capability_leaks(
    overrides: dict[str, Any], message: str
) -> None:
    """Strict mode fails fast when a conflicting capability is configured."""
    with pytest.raises(ValueError, match=message):
        AgenticApproach(**_strict_args(**overrides))


def test_strict_blackbox_supports_render_tools() -> None:
    """Strict mode advertises rendering without advertising withheld helpers."""
    approach = AgenticApproach(
        **_strict_args(mcp_tools=("render_state", "render_policy"))
    )
    prompt, system_prompt, _ = (
        approach._build_agentic_prompts()  # pylint: disable=protected-access
    )
    combined = prompt + system_prompt
    assert "mcp__robocode-tools__render_state" in combined
    assert "mcp__robocode-tools__render_policy" in combined
    assert "obs.tolist()" in combined
    assert "env.get_state" not in combined
    assert "env.observation_space.devectorize" not in combined
    assert "devectorize/vectorize to inspect" not in combined


def test_strict_server_allows_reset_step_and_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host rejects helper commands independently of client behavior."""
    env = _DummyEnv()
    registry = _HandleRegistry()
    monkeypatch.setattr(
        "robocode.utils.env_server_runtime._render_state",
        lambda *_args, **_kwargs: "mcp_renders/state.png",
    )
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
    rendered = _dispatch(
        env, {"cmd": "render_state", "seed": 4}, tmp_path, registry, strict=True
    )
    assert rendered["path"] == "mcp_renders/state.png"
    with pytest.raises(ValueError, match="only reset, step, and render_state"):
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


def test_strict_mcp_uses_separate_python_environment() -> None:
    """MCP startup must not add its dependencies to the generated-code Python."""
    assert STRICT_BLACKBOX_MCP_PYTHON != STRICT_BLACKBOX_PYTHON
    command = " ".join(
        _mcp_prestart_wrapper(["agent"], python_cmd=STRICT_BLACKBOX_PYTHON)
    )
    assert f"{STRICT_BLACKBOX_PYTHON} -c" in command


@requires_strict_docker
def test_strict_container_keeps_generated_python_dependency_clean() -> None:
    """The MCP installation must not alter the generated-program interpreter."""
    code = (
        "from importlib.util import find_spec; import numpy, scipy; "
        "assert find_spec('mcp') is None; assert find_spec('robocode') is None"
    )
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            STRICT_BLACKBOX_PYTHON,
            _STRICT_IMAGE,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@requires_strict_docker
def test_strict_container_mcp_renders_state_and_policy_through_host(
    tmp_path: Path,
) -> None:
    """The isolated MCP interpreter can proxy strict renders to the host."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "approach.py").write_text(
        "import numpy as np\n"
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        self._action_space = action_space\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return np.zeros(self._action_space.shape,"
        " dtype=self._action_space.dtype)\n"
    )
    env = KinderGeom2DEnv(_RENDER_ENV_CFG["env_id"])
    try:
        with env_server_running(json.dumps(_RENDER_ENV_CFG), sandbox, strict=True) as (
            port,
            token,
        ):
            write_env_spaces(
                sandbox,
                container_backend="docker",
                port=port,
                token=token,
                observation_space=env.observation_space,
                action_space=env.action_space,
                max_steps=5,
                strict=True,
            )
            code = (
                "import asyncio, json; from pathlib import Path; "
                "from robocode.mcp.server import build_blackbox_server; "
                "srv=build_blackbox_server(['render_state','render_policy'], "
                "Path('/sandbox/env_spaces.json')); "
                "_,state=asyncio.run(srv.call_tool('render_state', {'seed': 3})); "
                "_,policy=asyncio.run(srv.call_tool('render_policy', "
                "{'seed': 3, 'max_steps': 2})); "
                "print(json.dumps({'state': state['result'], "
                "'policy': policy['result']}))"
            )
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    "--entrypoint",
                    STRICT_BLACKBOX_MCP_PYTHON,
                    "-v",
                    f"{sandbox.resolve()}:/sandbox",
                    _STRICT_IMAGE,
                    "-c",
                    code,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
    finally:
        env.close()
    assert result.returncode == 0, result.stdout + result.stderr
    rendered = json.loads(result.stdout.strip().splitlines()[-1])
    state_path = Path(rendered["state"])
    assert state_path.name.startswith("state_seed3")
    assert (sandbox / "mcp_renders" / state_path.name).exists()
    assert rendered["policy"]
    for path in rendered["policy"]:
        assert (sandbox / "mcp_renders" / Path(path).name).exists()


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
        (
            {"approach.py": "import builtins\nbuiltins.__import__('os')\n"},
            "approach.py:2: __import__",
        ),
        (
            {"approach.py": "import sys\nenv = sys.modules['robocode']\n"},
            "approach.py:2: sys.modules",
        ),
        (
            {"approach.py": "import sys as s\ns.modules.get('robocode')\n"},
            "approach.py:2: sys.modules",
        ),
        ({"approach.py": "from sys import modules\n"}, "approach.py:1: sys.modules"),
        ({"approach.py": "from .. import helper\n"}, "relative import above"),
        ({"approach.py": "from env_client import make_env\n"}, "import env_client"),
    ],
)
def test_strict_imports_reject_everything_else(
    tmp_path: Path, files: dict[str, str], offender: str
) -> None:
    """Nested, sibling-transitive, dynamic, run-time, and client imports are named."""
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


def test_strict_loader_isolates_same_named_siblings_between_sandboxes(
    tmp_path: Path,
) -> None:
    """Sequential policies may reuse a sibling name without sharing its module."""
    entries: list[Path] = []
    for sandbox_name, value in (("first", 1), ("second", 2)):
        sandbox = tmp_path / sandbox_name
        _write(sandbox, "shared_policy_helper.py", f"VALUE = {value}\n")
        entries.append(
            _write(
                sandbox,
                "approach.py",
                "import shared_policy_helper\n"
                "class GeneratedApproach:\n"
                "    def __init__(self, *args, **kwargs):\n"
                "        self.value = shared_policy_helper.VALUE\n",
            )
        )

    space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    try:
        first = load_generated_approach(
            entries[0], space, space, {}, strict_imports=True
        )
        second = load_generated_approach(
            entries[1], space, space, {}, strict_imports=True
        )

        assert first.value == 1
        assert second.value == 2
        helper = sys.modules["shared_policy_helper"]
        assert helper.__file__ is not None
        assert Path(helper.__file__).is_relative_to(tmp_path / "second")
    finally:
        sys.modules.pop("shared_policy_helper", None)
