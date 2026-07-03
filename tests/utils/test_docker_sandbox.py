"""Tests for docker_sandbox.py.

Tests are split into two categories:

**Unit tests** (no Docker required)
    Verify config defaults, repo-root detection, and that
    :func:`_setup_sandbox_dir` creates the correct files on disk.

**Integration tests** (require the ``robocode-sandbox`` Docker image)
    Spin up the container with ``--entrypoint /bin/bash`` (bypassing the
    firewall init so no ``NET_ADMIN`` capability is needed) and assert that
    the agent's view matches expectations:

    * ``/sandbox/`` contains exactly the expected top-level entries.
    * ``/sandbox/primitives/`` holds the allowed primitive ``.py`` files.
    * ``/robocode/.venv/bin/python`` is executable and reports Python 3.11.
    * Key packages (numpy, gymnasium, robocode) are importable via the venv.
    * ``/robocode/third-party/kindergarden/`` reflects the bind-mounted submodule.
    * the bind-mounted ``kinder`` package is importable.
    * Robocode source is **not** present directly inside ``/sandbox/``.
    * The ``validate_sandbox.py`` hook blocks writes outside ``/sandbox``.

Integration tests are skipped automatically when the image is unavailable;
build it with ``bash docker/build.sh`` to run them.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.utils.backends import DEFAULT_BACKEND
from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    DockerSandboxConfig,
    _copy_src,
    _docker_run_prefix,
    _filtered_repo_mounts,
    _find_repo_root,
    _is_local_model,
    _setup_sandbox_dir,
)
from robocode.utils.env_server import (
    ENV_CLIENT_SRC,
    env_server_running,
    serialize_space,
)

_DOCKER_IMAGE = "robocode-sandbox"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_available() -> bool:
    """Return True if the robocode-sandbox Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", _DOCKER_IMAGE],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


requires_docker = pytest.mark.skipif(
    not _image_available(),
    reason=(
        f"Docker image '{_DOCKER_IMAGE}' not available; "
        "build it with: bash docker/build.sh"
    ),
)


def _run_in_container(
    sandbox_dir: Path,
    bash_cmd: str,
    *,
    uv_sync: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run *bash_cmd* in the container with sandbox, src, and kindergarden mounted.

    Uses ``--entrypoint /bin/bash`` to skip ``init-firewall.sh`` so that
    no ``NET_ADMIN`` capability is required during tests.

    When *uv_sync* is True, ``uv sync`` is run before *bash_cmd* so that
    the venv is available (since it is no longer baked into the image).
    """
    repo_root = _find_repo_root()

    # Create a filtered copy of src/ (oracles stripped).

    tmp_dir = tempfile.mkdtemp(prefix="robocode-test-src-")
    filtered_src = Path(tmp_dir) / "src"
    _copy_src(repo_root / "src", filtered_src)

    prefix = (
        "cd /robocode && uv sync --frozen --python python3.11 && cd /sandbox && "
        if uv_sync
        else ""
    )
    try:
        return subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "/bin/bash",
                "-v",
                f"{sandbox_dir.resolve()}:/sandbox",
                "-v",
                f"{filtered_src.resolve()}:/robocode/src",
                "-v",
                f"{(repo_root / 'third-party' / 'kindergarden').resolve()}"
                ":/robocode/third-party/kindergarden",
                "-w",
                "/sandbox",
                _DOCKER_IMAGE,
                "-c",
                prefix + bash_cmd,
            ],
            capture_output=True,
            text=True,
            timeout=300 if uv_sync else 60,
            check=False,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Unit tests — no Docker needed
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    """DockerSandboxConfig fields have the expected defaults."""
    config = DockerSandboxConfig(sandbox_dir=Path("/tmp/test"))
    assert config.docker_image == _DOCKER_IMAGE
    assert config.model == "sonnet"
    assert config.max_budget_usd == 5.0
    assert config.system_prompt == ""
    assert config.prompt == ""
    assert config.output_filename == ""
    assert not config.init_files
    assert not config.primitive_names
    # Docker reaches host-loopback services via the gateway alias; the base
    # SandboxConfig (local) and apptainer keep the loopback default.
    assert config.local_model_host == "host.docker.internal"


def test_find_repo_root_has_pyproject() -> None:
    """_find_repo_root() returns a directory containing pyproject.toml."""
    root = _find_repo_root()
    assert (root / "pyproject.toml").exists()


def test_filtered_repo_mounts_blackbox_excludes_env_source() -> None:
    """Blackbox mounts contain no environment source anywhere."""
    with _filtered_repo_mounts(blackbox=True) as (src, kindergarden, _):
        assert not (src / "robocode" / "environments").exists()
        assert not (src / "robocode" / "oracles").exists()
        assert not (kindergarden / "src" / "kinder" / "envs").exists()
        assert not (kindergarden / "demos").exists()
        # The package skeleton survives so the entrypoint's uv sync works.
        assert (src / "robocode" / "__init__.py").exists()
        assert (kindergarden / "pyproject.toml").exists()
        assert (kindergarden / "src" / "kinder" / "core.py").exists()


def test_filtered_repo_mounts_default_keeps_env_source() -> None:
    """Non-blackbox mounts keep the environment source."""
    with _filtered_repo_mounts() as (src, kindergarden, _):
        assert (src / "robocode" / "environments").exists()
        assert (kindergarden / "src" / "kinder" / "envs").exists()


def test_filtered_repo_mounts_excludes_bilevel_by_default() -> None:
    """Without include_bilevel, no kinder-baselines source is prepared (models OFF)."""
    with _filtered_repo_mounts() as (_src, _kg, kinder_baselines):
        assert kinder_baselines is None


def test_filtered_repo_mounts_includes_bilevel_when_requested() -> None:
    """With include_bilevel, only the two depended-on subpackages are copied."""
    with _filtered_repo_mounts(include_bilevel=True) as (_src, _kg, kinder_baselines):
        assert kinder_baselines is not None
        assert (
            kinder_baselines / "kinder-bilevel-planning" / "pyproject.toml"
        ).exists()
        assert (kinder_baselines / "kinder-models" / "pyproject.toml").exists()
        # the rest of the monorepo is not copied
        assert not (kinder_baselines / "kinder-rl").exists()


def test_docker_run_prefix_omits_bilevel_without_primitive(tmp_path: Path) -> None:
    """No kinder-baselines mount or extra-sync env when bilevel is not requested."""
    cmd = _docker_run_prefix(
        "c", "img", tmp_path, tmp_path / "src", tmp_path / "kg", None, [], []
    )
    joined = " ".join(cmd)
    assert "kinder-baselines" not in joined
    assert "ROBOCODE_UV_EXTRA_ARGS" not in joined


def test_docker_run_prefix_adds_bilevel_when_present(tmp_path: Path) -> None:
    """The kinder-baselines mount and `--extra bilevel` sync are added when
    requested."""
    kb = tmp_path / "kb"
    cmd = _docker_run_prefix(
        "c", "img", tmp_path, tmp_path / "src", tmp_path / "kg", kb, [], []
    )
    assert f"{kb.resolve()}:/robocode/third-party/kinder-baselines" in cmd
    assert "ROBOCODE_UV_EXTRA_ARGS=--extra bilevel" in cmd


def test_is_local_model() -> None:
    """Local model servers (ollama/vllm) are detected; remote API models are not."""
    assert _is_local_model("ollama/qwen3:0.6b")
    assert _is_local_model("vllm/Qwen/Qwen2.5-0.5B")
    assert not _is_local_model("openai/gpt-4o")
    assert not _is_local_model("anthropic/claude-sonnet-4-6")
    assert not _is_local_model("sonnet")


@requires_docker
def test_container_blackbox_no_env_source(tmp_path: Path) -> None:
    """With blackbox mounts, uv sync still works but env code is gone."""
    with _filtered_repo_mounts(blackbox=True) as (src, kindergarden, _):
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "/bin/bash",
                "-v",
                f"{tmp_path.resolve()}:/sandbox",
                "-v",
                f"{src.resolve()}:/robocode/src",
                "-v",
                f"{kindergarden.resolve()}:/robocode/third-party/kindergarden",
                "-w",
                "/sandbox",
                _DOCKER_IMAGE,
                "-c",
                "cd /robocode && uv sync --frozen --python python3.11 && "
                f"{DOCKER_PYTHON} -c 'import robocode.utils' && "
                # The blackbox MCP server must stay importable even though the
                # env source and primitives are gone (it proxies to the host).
                f"{DOCKER_PYTHON} -c 'import robocode.mcp.server' && "
                f"! {DOCKER_PYTHON} -c 'import robocode.environments' "
                "2>/dev/null && "
                f"! {DOCKER_PYTHON} -c 'import robocode.primitives' 2>/dev/null && "
                f"! {DOCKER_PYTHON} -c 'import kinder.envs' 2>/dev/null && "
                "! find / -name '*.py' 2>/dev/null "
                "| grep -E 'robocode/environments|kinder/envs' -q && "
                "echo BLACKBOX_OK",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    assert "BLACKBOX_OK" in result.stdout, result.stdout + result.stderr


# A battery of adversarial probes a "models OFF" agent might run to recover the
# bilevel planning models (predicates/operators/controllers). Each prints a
# `LEAK_<name>` marker ONLY if it succeeds; the test asserts none appear. Run in
# one container after a base `uv sync` (no `--extra bilevel`, no kinder-baselines
# mount), exactly as a real models-OFF sandbox is set up.
_MODELS_OFF_RED_TEAM_PROBES = r"""
PY=/robocode/.venv/bin/python
# 1. direct imports of the bilevel packages must all fail
for m in kinder_bilevel_planning kinder_models bilevel_planning; do
    $PY -c "import $m" 2>/dev/null && echo "LEAK_import_$m"
done
# 2. the lazy model-builder must raise (ImportError), not return models
$PY - <<'EOF' 2>/dev/null && echo "LEAK_build_models"
import types
from robocode.utils.bilevel import build_sesame_models
e = types.SimpleNamespace(bilevel_env_name="obstruction2d",
                          bilevel_env_model_kwargs={},
                          observation_space=None, action_space=None)
build_sesame_models(e)
EOF
# 3. no kinder-baselines source dir is present
test -e /robocode/third-party/kinder-baselines && echo "LEAK_kb_dir"
# 4. the REAL model source (definitions, which live only in the bilevel packages)
# is not reachable. We match definitions, not robocode's own references to them.
grep -rEl "def create_bilevel_planning_models|def state_abstractor|class SesameModels" \
    /robocode 2>/dev/null | head -1 | grep -q . && echo "LEAK_model_source"
find /robocode -name '*.py' 2>/dev/null \
    | grep -E "kinder_bilevel_planning|kinder_models|/bilevel_planning/" \
    | head -1 | grep -q . && echo "LEAK_source_files"
# 5. the agent cannot self-serve the models from PyPI (not published there)
$PY -m pip install --quiet kinder_bilevel_planning 2>/dev/null \
    && echo "LEAK_pip_install"
# 6. the prompt-descriptions module is not shipped, so the agent cannot read the
# description (symbolic API, example atoms) of a primitive it was not granted
test -e /robocode/src/robocode/primitive_descriptions.py && echo "LEAK_descriptions_module"
grep -rE "symbolic predicates and operators|get_abstract_state\(obs\)" /robocode/src \
    2>/dev/null | head -1 | grep -q . && echo "LEAK_description_text"
echo PROBES_DONE
"""


@requires_docker
def test_container_models_off_red_team(tmp_path: Path) -> None:
    """Red-team: a models-OFF sandbox yields no path to the bilevel models.

    Runs a battery of adversarial probes (import, lazy builder, filesystem search,
    symbol grep, self-install) in a container set up exactly as a real models-OFF
    run (base sync, no kinder-baselines mount). None may recover the models, while
    `robocode.utils` (holding the lazy builder) stays importable.
    """
    with _filtered_repo_mounts() as (src, kindergarden, kinder_baselines):
        assert kinder_baselines is None
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "/bin/bash",
                "-v",
                f"{tmp_path.resolve()}:/sandbox",
                "-v",
                f"{src.resolve()}:/robocode/src",
                "-v",
                f"{kindergarden.resolve()}:/robocode/third-party/kindergarden",
                "-w",
                "/sandbox",
                _DOCKER_IMAGE,
                "-c",
                "cd /robocode && uv sync --frozen --python python3.11 && "
                f"{DOCKER_PYTHON} -c 'import robocode.utils' && "
                + _MODELS_OFF_RED_TEAM_PROBES,
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    out = result.stdout + result.stderr
    assert "PROBES_DONE" in out, out
    leaks = [line for line in out.splitlines() if line.startswith("LEAK_")]
    assert not leaks, f"models-OFF isolation breached: {leaks}\n{out}"


@requires_docker
def test_container_blackbox_render_proxy(tmp_path: Path) -> None:
    """A blackbox container renders via the host env server over TCP.

    Exercises the same cross-container path the MCP render tools use: the
    in-container env_client connects to the host env server, which renders
    and writes the PNG into the bind-mounted sandbox dir.
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    shutil.copy2(ENV_CLIENT_SRC, sandbox / "env_client.py")
    env_cfg = json.dumps(
        {
            "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
            "env_id": "kinder/Motion2D-p0-v0",
        }
    )
    with env_server_running(env_cfg, sandbox) as (port, token):
        env = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
        (sandbox / "env_spaces.json").write_text(
            json.dumps(
                {
                    "host": "host.docker.internal",
                    "port": port,
                    "token": token,
                    "observation_space": serialize_space(env.observation_space),
                    "action_space": serialize_space(env.action_space),
                    "max_steps": 5,
                }
            )
        )
        env.close()
        with _filtered_repo_mounts(blackbox=True) as (src, kindergarden, _):
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    "--entrypoint",
                    "/bin/bash",
                    "-v",
                    f"{sandbox.resolve()}:/sandbox",
                    "-v",
                    f"{src.resolve()}:/robocode/src",
                    "-v",
                    f"{kindergarden.resolve()}:/robocode/third-party/kindergarden",
                    "-w",
                    "/sandbox",
                    _DOCKER_IMAGE,
                    "-c",
                    "cd /robocode && uv sync --frozen --python python3.11 "
                    ">/dev/null 2>&1 && cd /sandbox && "
                    f'{DOCKER_PYTHON} -c "from env_client import make_env; '
                    'print(make_env().render_state(seed=0))"',
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
    assert result.returncode == 0, result.stdout + result.stderr
    rel = result.stdout.strip().splitlines()[-1]
    assert (sandbox / rel).exists(), f"render output {rel!r} missing"


@requires_docker
def test_container_blackbox_render_policy_runs_in_sandbox(tmp_path: Path) -> None:
    """render_policy runs approach.py inside the blackbox container.

    The policy episode executes in the container (which has no env source); only the
    per-state render crosses to the host. Confirms the frames come back without the host
    ever executing approach.py.
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    shutil.copy2(ENV_CLIENT_SRC, sandbox / "env_client.py")
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
    env_cfg = json.dumps(
        {
            "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
            "env_id": "kinder/Motion2D-p0-v0",
        }
    )
    with env_server_running(env_cfg, sandbox) as (port, token):
        env = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
        (sandbox / "env_spaces.json").write_text(
            json.dumps(
                {
                    "host": "host.docker.internal",
                    "port": port,
                    "token": token,
                    "observation_space": serialize_space(env.observation_space),
                    "action_space": serialize_space(env.action_space),
                    "max_steps": 5,
                }
            )
        )
        env.close()
        with _filtered_repo_mounts(blackbox=True) as (src, kindergarden, _):
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    "--entrypoint",
                    "/bin/bash",
                    "-v",
                    f"{sandbox.resolve()}:/sandbox",
                    "-v",
                    f"{src.resolve()}:/robocode/src",
                    "-v",
                    f"{kindergarden.resolve()}:/robocode/third-party/kindergarden",
                    "-w",
                    "/sandbox",
                    _DOCKER_IMAGE,
                    "-c",
                    "cd /robocode && uv sync --frozen --python python3.11 "
                    ">/dev/null 2>&1 && cd /sandbox && "
                    f'{DOCKER_PYTHON} -c "from env_client import make_env; '
                    "paths = make_env().render_policy(seed=0, max_steps=3); "
                    'print(paths[-1])"',
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
    assert result.returncode == 0, result.stdout + result.stderr
    rel = result.stdout.strip().splitlines()[-1]
    assert (sandbox / rel).exists(), f"render output {rel!r} missing"


def test_setup_creates_sandbox_dir(tmp_path: Path) -> None:
    """_setup_sandbox_dir() creates the sandbox directory if absent."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    assert not config.sandbox_dir.exists()
    _setup_sandbox_dir(config)
    assert config.sandbox_dir.is_dir()


def test_setup_creates_claude_md(tmp_path: Path) -> None:
    """CLAUDE.md is created and references the Docker Python path."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    DEFAULT_BACKEND.setup_sandbox_files(config, docker_python=DOCKER_PYTHON)
    claude_md = config.sandbox_dir / "CLAUDE.md"
    assert claude_md.exists()
    assert DOCKER_PYTHON in claude_md.read_text()


def test_setup_creates_dot_claude_settings(tmp_path: Path) -> None:
    """.claude/settings.json is created with the PreToolUse hook."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    DEFAULT_BACKEND.setup_sandbox_files(config)
    settings = config.sandbox_dir / ".claude" / "settings.json"
    assert settings.exists()
    assert "PreToolUse" in settings.read_text()


def test_setup_creates_validate_script(tmp_path: Path) -> None:
    """.claude/validate_sandbox.py is created."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    DEFAULT_BACKEND.setup_sandbox_files(config)
    assert (config.sandbox_dir / ".claude" / "validate_sandbox.py").exists()


def test_setup_creates_git_repo(tmp_path: Path) -> None:
    """_setup_sandbox_dir() initialises a git repo inside the sandbox."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    assert (config.sandbox_dir / ".git").is_dir()


def test_setup_copies_only_requested_primitives(tmp_path: Path) -> None:
    """Only the requested primitive .py files are copied into sandbox/primitives/."""
    config = DockerSandboxConfig(
        sandbox_dir=tmp_path / "sandbox",
        primitive_names=("csp", "check_action_collision"),
    )
    _setup_sandbox_dir(config)
    primitives_dir = config.sandbox_dir / "primitives"
    assert primitives_dir.is_dir()
    copied = {f.name for f in primitives_dir.glob("*.py")}
    assert copied == {"csp.py", "check_action_collision.py"}


def test_setup_blackbox_skips_env_dependent_primitives(tmp_path: Path) -> None:
    """Blackbox setup omits env-dependent primitive source, keeps generic ones.

    check_action_collision imports the (hidden) env, so its source cannot be copied into
    a blackbox sandbox; the sandbox builds it as a host proxy via
    env_client.make_primitives instead. Generic primitives are still copied.
    """
    config = DockerSandboxConfig(
        sandbox_dir=tmp_path / "sandbox",
        primitive_names=("csp", "check_action_collision", "BiRRT"),
        blackbox=True,
    )
    _setup_sandbox_dir(config)
    copied = {f.name for f in (config.sandbox_dir / "primitives").glob("*.py")}
    assert copied == {"csp.py", "motion_planning.py"}
    assert "check_action_collision.py" not in copied


def test_setup_no_primitives_dir_when_none_requested(tmp_path: Path) -> None:
    """No primitives/ directory is created when primitive_names is empty."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    assert not (config.sandbox_dir / "primitives").exists()


def test_setup_copies_init_files(tmp_path: Path) -> None:
    """Caller-supplied init_files are copied into the sandbox."""
    source = tmp_path / "hello.txt"
    source.write_text("hello world")
    config = DockerSandboxConfig(
        sandbox_dir=tmp_path / "sandbox",
        init_files={"subdir/hello.txt": source},
    )
    _setup_sandbox_dir(config)
    dest = config.sandbox_dir / "subdir" / "hello.txt"
    assert dest.exists()
    assert dest.read_text() == "hello world"


def test_setup_is_idempotent(tmp_path: Path) -> None:
    """Calling _setup_sandbox_dir twice raises no errors."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    _setup_sandbox_dir(config)


# ---------------------------------------------------------------------------
# Integration tests — require the robocode-sandbox Docker image
# ---------------------------------------------------------------------------


@requires_docker
def test_container_cwd_is_sandbox(tmp_path: Path) -> None:
    """The container's working directory is /sandbox."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(config.sandbox_dir, "pwd")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/sandbox"


@requires_docker
def test_container_sandbox_top_level_entries(tmp_path: Path) -> None:
    """The sandbox contains exactly the expected top-level entries and no extras.

    Expected entries: primitives/, CLAUDE.md, .claude/, .git/ Must NOT
    contain: src/, pyproject.toml, third-party/ (host repo files)
    """
    config = DockerSandboxConfig(
        sandbox_dir=tmp_path / "sandbox",
        primitive_names=("csp", "check_action_collision"),
    )
    _setup_sandbox_dir(config)
    DEFAULT_BACKEND.setup_sandbox_files(config, docker_python=DOCKER_PYTHON)
    result = _run_in_container(config.sandbox_dir, "ls -a /sandbox")
    assert result.returncode == 0, result.stderr
    listed = set(result.stdout.split())

    # Expected entries.
    assert "primitives" in listed
    assert "CLAUDE.md" in listed
    assert ".claude" in listed
    assert ".git" in listed

    # Host repo files must NOT bleed into the sandbox.
    assert "pyproject.toml" not in listed
    assert "src" not in listed
    assert "third-party" not in listed


@requires_docker
def test_container_primitives_files(tmp_path: Path) -> None:
    """The primitive .py files inside /sandbox/primitives/ match the requested ones."""
    config = DockerSandboxConfig(
        sandbox_dir=tmp_path / "sandbox",
        primitive_names=("check_action_collision", "BiRRT"),
    )
    _setup_sandbox_dir(config)
    result = _run_in_container(config.sandbox_dir, "ls /sandbox/primitives/")
    assert result.returncode == 0, result.stderr
    listed = set(result.stdout.split())
    assert listed == {"check_action_collision.py", "motion_planning.py"}


@requires_docker
def test_container_venv_python_exists(tmp_path: Path) -> None:
    """The venv Python interpreter exists and is executable after uv sync."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir, f"{DOCKER_PYTHON} --version", uv_sync=True
    )
    assert result.returncode == 0, result.stderr
    version_output = result.stdout + result.stderr  # Python prints to stderr on 3.x
    assert "Python 3.11" in version_output


@requires_docker
def test_container_venv_imports_core_packages(tmp_path: Path) -> None:
    """Numpy, gymnasium, and robocode are importable via the venv Python."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        f"{DOCKER_PYTHON} -c 'import numpy, gymnasium, robocode; print(\"OK\")'",
        uv_sync=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@requires_docker
def test_container_kindergarden_mounted(tmp_path: Path) -> None:
    """/robocode/third-party/kindergarden/ is present (bind-mounted submodule)."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        "test -d /robocode/third-party/kindergarden && echo mounted || echo absent",
    )
    assert result.returncode == 0, result.stderr
    assert "mounted" in result.stdout


@requires_docker
def test_container_kindergarden_importable(tmp_path: Path) -> None:
    """The bind-mounted kinder package is importable via the venv."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        f"{DOCKER_PYTHON} -c 'import kinder; print(\"OK\")'",
        uv_sync=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@requires_docker
def test_container_oracles_not_present(tmp_path: Path) -> None:
    """Oracle solutions must NOT be present in the container."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        "test -d /robocode/src/robocode/oracles && echo found || echo absent",
    )
    assert "absent" in result.stdout


@requires_docker
def test_container_robocode_source_not_in_sandbox(tmp_path: Path) -> None:
    """Robocode source is NOT present inside /sandbox/ directly."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        "test -d /sandbox/robocode && echo found || echo absent",
    )
    assert "absent" in result.stdout


@requires_docker
def test_container_hook_blocks_write_outside_sandbox(tmp_path: Path) -> None:
    """validate_sandbox.py denies a Write tool call to a path outside /sandbox."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    DEFAULT_BACKEND.setup_sandbox_files(config)
    result = _run_in_container(
        config.sandbox_dir,
        'echo \'{"tool_name": "Write", "tool_input": {"file_path": "/etc/evil"}}\' '
        "| python3 /sandbox/.claude/validate_sandbox.py",
    )
    assert "deny" in result.stdout


@requires_docker
def test_container_hook_allows_write_inside_sandbox(tmp_path: Path) -> None:
    """validate_sandbox.py allows a Write tool call to a path inside /sandbox."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    DEFAULT_BACKEND.setup_sandbox_files(config)
    bash_cmd = (
        'echo \'{"tool_name": "Write", '
        '"tool_input": {"file_path": "/sandbox/approach.py"}}\''
        " | python3 /sandbox/.claude/validate_sandbox.py; echo exit=$?"
    )
    result = _run_in_container(config.sandbox_dir, bash_cmd)
    assert "exit=0" in result.stdout
    assert "deny" not in result.stdout


@requires_docker
def test_container_files_persist_on_host(tmp_path: Path) -> None:
    """Files written inside /sandbox (including subdirectories) are visible on the host
    via the bind-mount."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        "echo 'print(42)' > /sandbox/approach.py && "
        "mkdir -p /sandbox/utils && "
        "echo 'X = 1' > /sandbox/helper.py && "
        "echo 'Y = 2' > /sandbox/utils/lib.py",
    )
    assert result.returncode == 0, result.stderr

    sandbox_dir = config.sandbox_dir
    assert (sandbox_dir / "approach.py").exists()
    assert (sandbox_dir / "helper.py").exists()
    assert (sandbox_dir / "utils" / "lib.py").exists()
    assert "print(42)" in (sandbox_dir / "approach.py").read_text()
    assert "X = 1" in (sandbox_dir / "helper.py").read_text()
    assert "Y = 2" in (sandbox_dir / "utils" / "lib.py").read_text()
