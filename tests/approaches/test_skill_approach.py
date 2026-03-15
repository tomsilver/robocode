"""Tests for SkillAgenticApproach sandbox setup.

Verifies that the sandbox is populated correctly with the skills
directory, initial approach files, and state directories, and that a
DummySkill subclass can be instantiated inside the Docker container.
"""

import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    DockerSandboxConfig,
    _docker_cmd_prefix,
    _find_repo_root,
    _setup_sandbox_dir,
)

_DOCKER_IMAGE = "robocode-sandbox"

_SKILLS_SRC = Path(__file__).resolve().parents[2] / "src" / "robocode" / "skills"
_INITIAL_SKILL_DIR = _SKILLS_SRC / "pushpullhook2d"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_available() -> bool:
    try:
        result = subprocess.run(
            [*_docker_cmd_prefix(), "image", "inspect", _DOCKER_IMAGE],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


requires_docker = pytest.mark.skipif(
    not _image_available(),
    reason=f"Docker image '{_DOCKER_IMAGE}' not available",
)


def _run_in_container(
    sandbox_dir: Path, bash_cmd: str
) -> subprocess.CompletedProcess[str]:
    """Run *bash_cmd* inside the container with sandbox and prpl-mono mounted."""
    repo_root = _find_repo_root()
    return subprocess.run(
        [
            *_docker_cmd_prefix(),
            "run",
            "--rm",
            "--entrypoint",
            "/bin/bash",
            "-v",
            f"{sandbox_dir.resolve()}:/sandbox",
            "-v",
            f"{(repo_root / 'prpl-mono').resolve()}:/robocode/prpl-mono:ro",
            "-w",
            "/sandbox",
            _DOCKER_IMAGE,
            "-c",
            bash_cmd,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _setup_skill_sandbox(tmp_path: Path) -> Path:
    """Create a sandbox dir populated like SkillAgenticApproach.train() does."""
    sandbox_dir = tmp_path / "sandbox"

    # Let DockerSandboxConfig create the standard scaffolding.
    config = DockerSandboxConfig(sandbox_dir=sandbox_dir)
    _setup_sandbox_dir(config)

    # Copy the entire skills directory (same as train() does).
    skills_dest = sandbox_dir / "skills"
    shutil.copytree(
        _SKILLS_SRC,
        skills_dest,
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    # Copy the initial approach .py files to sandbox root.
    for py_file in sorted(_INITIAL_SKILL_DIR.glob("*.py")):
        shutil.copy2(py_file, sandbox_dir / py_file.name)

    return sandbox_dir


# ---------------------------------------------------------------------------
# Unit tests — no Docker needed
# ---------------------------------------------------------------------------


def test_sandbox_has_skills_directory(tmp_path: Path) -> None:
    """The skills/ directory is copied into the sandbox."""
    sandbox_dir = _setup_skill_sandbox(tmp_path)
    skills_dir = sandbox_dir / "skills"
    assert skills_dir.is_dir()
    assert (skills_dir / "utils.py").exists()
    assert (skills_dir / "pushpullhook2d").is_dir()
    assert (skills_dir / "pushpullhook2d" / "approach.py").exists()


def test_sandbox_has_initial_skill_files_at_root(tmp_path: Path) -> None:
    """Initial skill .py files are copied to the sandbox root."""
    sandbox_dir = _setup_skill_sandbox(tmp_path)
    assert (sandbox_dir / "approach.py").exists()
    assert (sandbox_dir / "pick_skill.py").exists()
    assert (sandbox_dir / "push_skill.py").exists()


def test_sandbox_state_dirs_populated(tmp_path: Path) -> None:
    """Failed/success state .npy files are copied into the sandbox."""
    sandbox_dir = _setup_skill_sandbox(tmp_path)

    # Create mock state dirs with dummy .npy files.
    failed_dir = tmp_path / "failed_states_src"
    failed_dir.mkdir()
    np.save(failed_dir / "init_episode_3.npy", {"dummy": 1})
    np.save(failed_dir / "init_episode_7.npy", {"dummy": 2})

    success_dir = tmp_path / "success_states_src"
    success_dir.mkdir()
    np.save(success_dir / "init_episode_0.npy", {"dummy": 3})

    # Mimic what train() does: copy into sandbox.
    dst_failed = sandbox_dir / "failed_states"
    dst_failed.mkdir()
    for f in sorted(failed_dir.glob("*.npy")):
        shutil.copy2(f, dst_failed / f.name)

    dst_success = sandbox_dir / "success_states"
    dst_success.mkdir()
    for f in sorted(success_dir.glob("*.npy")):
        shutil.copy2(f, dst_success / f.name)

    assert (dst_failed / "init_episode_3.npy").exists()
    assert (dst_failed / "init_episode_7.npy").exists()
    assert (dst_success / "init_episode_0.npy").exists()


# ---------------------------------------------------------------------------
# Integration tests — require Docker
# ---------------------------------------------------------------------------


@requires_docker
def test_container_skills_directory_accessible(tmp_path: Path) -> None:
    """The skills/ directory is visible inside the container."""
    sandbox_dir = _setup_skill_sandbox(tmp_path)
    result = _run_in_container(sandbox_dir, "ls /sandbox/skills/")
    assert result.returncode == 0, result.stderr
    listed = set(result.stdout.split())
    assert "utils.py" in listed
    assert "pushpullhook2d" in listed


@requires_docker
def test_container_skills_utils_importable(tmp_path: Path) -> None:
    """skills/utils.py is importable from the sandbox."""
    sandbox_dir = _setup_skill_sandbox(tmp_path)
    result = _run_in_container(
        sandbox_dir,
        f"{DOCKER_PYTHON} -c '"
        "import sys; sys.path.insert(0, \"/sandbox\"); "
        "from skills.utils import TrajectorySamplingFailure, "
        "run_motion_planning_for_crv_robot; "
        "print(\"OK\")"
        "'",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "OK" in result.stdout


@requires_docker
def test_container_initial_skill_importable(tmp_path: Path) -> None:
    """The initial pick_skill.py is importable from the sandbox root."""
    sandbox_dir = _setup_skill_sandbox(tmp_path)
    result = _run_in_container(
        sandbox_dir,
        f"{DOCKER_PYTHON} -c '"
        "import sys; sys.path.insert(0, \"/sandbox\"); "
        "from pick_skill import GroundPickController; "
        "print(GroundPickController.__name__)"
        "'",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "GroundPickController" in result.stdout


@requires_docker
def test_container_dummy_skill_works(tmp_path: Path) -> None:
    """A DummySkill subclass can be defined and instantiated inside Docker."""
    sandbox_dir = _setup_skill_sandbox(tmp_path)

    # Write a DummySkill that imports from the skills dir and kinder.
    dummy_script = textwrap.dedent("""\
        import sys
        sys.path.insert(0, "/sandbox")

        from kinder_models.kinematic2d.utils import Kinematic2dRobotController
        from skills.utils import TrajectorySamplingFailure
        from kinder.envs.kinematic2d.structs import SE2Pose

        class DummySkill(Kinematic2dRobotController):
            def sample_parameters(self, x, rng):
                return (0.5,)

            def _get_vacuum_actions(self):
                return 0.0, 0.0

            def _generate_waypoints(self, state):
                return [(SE2Pose(0.0, 0.0, 0.0), 0.1)]

        # Verify the class hierarchy works.
        assert issubclass(DummySkill, Kinematic2dRobotController)
        print(f"DummySkill MRO: {[c.__name__ for c in DummySkill.__mro__]}")
        print("PASS")
    """)
    (sandbox_dir / "test_dummy_skill.py").write_text(dummy_script)

    result = _run_in_container(
        sandbox_dir,
        f"{DOCKER_PYTHON} /sandbox/test_dummy_skill.py",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "PASS" in result.stdout
    assert "DummySkill" in result.stdout


@requires_docker
def test_container_npy_state_loadable(tmp_path: Path) -> None:
    """An ObjectCentricState saved as .npy can be loaded inside Docker."""
    sandbox_dir = _setup_skill_sandbox(tmp_path)

    # Save a real ObjectCentricState from the env.
    from kinder.envs.kinematic2d.pushpullhook2d import (  # pylint: disable=import-outside-toplevel
        ObjectCentricPushPullHook2DEnv,
    )

    env = ObjectCentricPushPullHook2DEnv(render_mode="rgb_array")
    state, _ = env.reset(seed=42)
    env.close()

    states_dir = sandbox_dir / "failed_states"
    states_dir.mkdir()
    np.save(states_dir / "init_episode_42.npy", state)

    load_script = textwrap.dedent("""\
        import numpy as np
        state = np.load(
            "failed_states/init_episode_42.npy", allow_pickle=True
        ).item()
        print(f"Type: {type(state).__name__}")
        print(f"Objects: {[o.name for o in state]}")

        from kinder.envs.kinematic2d.pushpullhook2d import (
            ObjectCentricPushPullHook2DEnv,
        )
        env = ObjectCentricPushPullHook2DEnv(
            render_mode="rgb_array", allow_state_access=True
        )
        env.reset()
        env.unwrapped.set_state(state)
        restored = env.unwrapped.get_state()
        robot = [o for o in restored if o.name == "robot"][0]
        print(f"Robot x: {restored.get(robot, 'x'):.6f}")
        env.close()
        print("PASS")
    """)
    (sandbox_dir / "test_load_state.py").write_text(load_script)

    result = _run_in_container(
        sandbox_dir,
        f"{DOCKER_PYTHON} /sandbox/test_load_state.py",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "PASS" in result.stdout
    assert "ObjectCentricState" in result.stdout
