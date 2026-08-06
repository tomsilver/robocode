"""Evaluate each sandbox commit in the background as the agent creates it.

Watches the sandbox git repo during a run and scores each new approach.py commit over
the run's eval suite, so the timeline is ready when the run ends instead of as a slow
post-run pass. Each version is exported and scored in its own subprocess (loading
generated code mutates sys.modules and episode workers fork, so in-thread is unsafe);
one eval at a time, drained before the primary evaluation. The sandbox is read-only.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from robocode.utils.approach_history import Snapshot, _export_snapshot, get_snapshots

logger = logging.getLogger(__name__)

_POLL_SECONDS = 5.0
# Head-room over the suite's own budget (num_eval_tasks * eval_timeout), to absorb
# startup/import before a revision that hangs there is killed.
_EVAL_STARTUP_MARGIN_SECONDS = 600.0
_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/robocode/utils/... -> repo root
_RUNNER = "experiments/run_experiment.py"

# Protect $1 read-only in a mount namespace, then drop the mount capability before
# exec'ing $2.. so the untrusted eval cannot undo it. See _isolate.
_ISOLATE_SETUP = (
    'mount --bind "$1" "$1" && mount -o remount,ro,bind "$1" && shift && '
    'exec setpriv --bounding-set=-all --no-new-privs -- "$@"'
)


def _isolate(cmd: list[str], protect_dir: Path) -> list[str]:
    """Wrap *cmd* so *protect_dir* is read-only and cannot be made writable again.

    A concurrent eval runs the untrusted policy on the host; without this it could
    write held-out rewards/seeds into the live training sandbox, which is bind-mounted
    into the still-running agent. The mount namespace makes that directory read-only
    for the eval, and the dropped CAP_SYS_ADMIN stops the policy remounting it. Needs
    an unprivileged user namespace (see :func:`_isolation_available`).
    """
    return [
        "unshare",
        "--user",
        "--map-root-user",
        "--mount",
        "bash",
        "-c",
        _ISOLATE_SETUP,
        "_",
        str(protect_dir),
        *cmd,
    ]


@functools.cache
def _isolation_available() -> bool:
    """Whether the isolation wrapper actually works here (probes the real prefix).

    Unprivileged user namespaces are blocked on CI runners and hardened kernels;
    there live eval must stay off rather than run an unisolated policy beside the agent.
    """
    if sys.platform != "linux":
        return False
    with tempfile.TemporaryDirectory() as probe_dir:
        try:
            result = subprocess.run(
                _isolate(["true"], Path(probe_dir)),
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
    return result.returncode == 0


def _child_env() -> dict[str, str]:
    """Eval-subprocess env: no telemetry sink, native thread pools capped to one."""
    env = dict(os.environ)
    env.pop("ROBOCODE_TELEMETRY", None)
    env.pop("ROBOCODE_RUN_ID", None)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"  # override an inherited high value, not just fill a default
    return env


def _eval_command(
    version_dir: Path, task_overrides: list[str], eval_seed: int
) -> list[str]:
    """Standalone-eval invocation for one exported commit.

    Re-passes the parent's overrides and pins eval_seed so versions are comparable.
    """
    return [
        sys.executable,
        _RUNNER,
        *task_overrides,
        f"approach.load_dir={version_dir}",
        f"hydra.run.dir={version_dir}",
        f"eval_seed={eval_seed}",
        "live_approach_history=false",  # never recurse
        "record_approach_history=false",
        "render_videos=false",
    ]


def _run_eval_subprocess(
    cmd: list[str], env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run one commit's standalone eval. A named seam so tests can intercept it.

    Bounded so a revision that hangs before the per-episode timeout (e.g. an
    approach.py that blocks on import) cannot make the watcher block the run forever;
    on timeout the child is killed and the version is recorded as failed.
    """
    try:
        return subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd, returncode=-1, stdout="", stderr=f"killed: exceeded {timeout}s"
        )


class _CommitEvalWatcher:
    """Polls the sandbox git repo and evaluates each new approach.py commit."""

    def __init__(
        self,
        sandbox_dir: Path,
        output_dir: Path,
        task_overrides: list[str],
        eval_seed: int,
        num_eval_tasks: int,
        eval_timeout: float,
    ) -> None:
        self._sandbox_dir = sandbox_dir
        self._history_dir = output_dir / "approach_history"
        self._task_overrides = task_overrides
        self._eval_seed = eval_seed
        # Ceiling per eval: the suite's own budget plus start-up head-room. A full
        # suite finishes within this; a revision hanging before its episodes do not.
        self._timeout = num_eval_tasks * eval_timeout + _EVAL_STARTUP_MARGIN_SECONDS
        # commit hash -> version id, assigned once in discovery order so a rewritten
        # history (amend/reset) never reuses a prior commit's vNNN directory.
        self._versions: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="commit-eval-watcher", daemon=True
        )

    def start(self) -> None:
        """Begin polling the sandbox repo for new commits."""
        self._thread.start()

    def stop(self) -> None:
        """Stop discovery and drain the backlog, so no eval overlaps the primary one."""
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._scan_and_eval()
            self._stop.wait(_POLL_SECONDS)
        self._scan_and_eval()  # final drain: catch commits after the last poll

    def _scan_and_eval(self) -> None:
        if not (self._sandbox_dir / ".git").exists():
            return  # sandbox not initialized yet
        try:
            snapshots = get_snapshots(self._sandbox_dir)
        except subprocess.CalledProcessError:
            return  # unreadable mid-commit; caught next scan
        for snap in snapshots:
            if snap.commit_hash in self._versions:
                continue
            version = len(self._versions)
            self._versions[snap.commit_hash] = version
            self._evaluate(snap, version)

    def _evaluate(self, snap: Snapshot, version: int) -> None:
        version_dir = self._history_dir / f"v{version:03d}"
        export = version_dir / "sandbox"
        if export.exists():
            shutil.rmtree(export)  # never mix a prior export's files into this one
        _export_snapshot(self._sandbox_dir, snap.commit_hash, export)
        logger.info(
            "live timeline: evaluating v%03d (%s) %r",
            version,
            snap.commit_hash[:8],
            snap.message,
        )
        cmd = _eval_command(version_dir, self._task_overrides, self._eval_seed)
        result = _run_eval_subprocess(
            _isolate(cmd, self._sandbox_dir), _child_env(), self._timeout
        )
        if result.returncode != 0:
            logger.warning(
                "live timeline: v%03d eval exited %d; see tail:\n%s",
                version,
                result.returncode,
                result.stderr[-800:],
            )
            self._write_failure(snap, version_dir, result.stderr)

    def _write_failure(self, snap: Snapshot, version_dir: Path, stderr: str) -> None:
        """Record a failed eval so the timeline has an entry for a broken commit.

        The child exits before writing results.json when approach.py fails to load, so
        without this a bad intermediate commit (expected during training) leaves a gap.
        """
        results = version_dir / "results.json"
        if results.exists():
            return  # the child wrote its own results; do not clobber them
        version_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "commit_hash": snap.commit_hash,
            "timestamp": snap.timestamp,
            "message": snap.message,
            "solve_rate": None,
            "error": stderr[-800:],
        }
        results.write_text(json.dumps(record, indent=2), encoding="utf-8")


@contextmanager
def live_commit_eval(
    *,
    enabled: bool,
    sandbox_dir: Path,
    output_dir: Path,
    task_overrides: list[str],
    eval_seed: int,
    num_eval_tasks: int,
    eval_timeout: float,
) -> Iterator[None]:
    """Score every new sandbox commit in the background for the duration of the block.

    A no-op when *enabled* is false. Fails closed when the sandbox cannot be isolated
    from the eval, since an unisolated concurrent eval could leak held-out data to the
    agent. Leaves a per-commit timeline under output_dir/approach_history/.
    """
    if not enabled:
        yield
        return
    if not _isolation_available():
        logger.warning(
            "live_approach_history is on but the training sandbox cannot be isolated "
            "(unprivileged user namespaces unavailable); disabling it rather than run "
            "an unisolated eval beside the agent, which could leak held-out data."
        )
        yield
        return
    watcher = _CommitEvalWatcher(
        sandbox_dir, output_dir, task_overrides, eval_seed, num_eval_tasks, eval_timeout
    )
    watcher.start()
    try:
        yield
    finally:
        watcher.stop()
