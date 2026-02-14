"""Test that an environment can be copied into a sandbox and run there."""

import inspect
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import robocode
from robocode.environments.base_env import BaseEnv
from robocode.environments.maze_env import MazeEnv
from robocode.utils.source_deps import collect_local_deps


def test_env_runs_in_sandbox(tmp_path: Path) -> None:
    """Copy MazeEnv source into a sandbox and run it in a subprocess."""
    sandbox = tmp_path / "sandbox"

    env_source = Path(inspect.getfile(MazeEnv))
    assert robocode.__file__ is not None
    pkg_root = Path(robocode.__file__).parent.parent

    deps = collect_local_deps(env_source, pkg_root)
    assert Path(inspect.getfile(BaseEnv)).resolve() in deps

    # Copy each file into the sandbox, preserving package structure.
    for dep in deps:
        rel = dep.relative_to(pkg_root)
        dest = sandbox / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dep, dest)

    # Create __init__.py files so the package is importable.
    for dirpath in sandbox.rglob("*"):
        if dirpath.is_dir() and any(dirpath.glob("*.py")):
            init = dirpath / "__init__.py"
            if not init.exists():
                init.touch()

    # Run a script in a subprocess that imports and exercises the env.
    script = sandbox / "run_env.py"
    script.write_text(textwrap.dedent("""\
        import sys
        sys.path.insert(0, ".")

        from robocode.environments.maze_env import MazeEnv

        env = MazeEnv(5, 8, 5, 8)
        state, info = env.reset(seed=42)
        assert state is not None

        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        assert next_state is not None
        assert reward == -1

        next_state_2 = env.sample_next_state(state, action, env.np_random)
        assert next_state_2 is not None

        print("OK")
    """))

    result = subprocess.run(
        [sys.executable, "run_env.py"],
        cwd=str(sandbox),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout
