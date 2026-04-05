"""Minimal episode test for the ClutteredRetrieval2D medium oracle approach."""

from pathlib import Path

import kinder
import pytest
from gymnasium.wrappers import RecordVideo
from numpy.typing import NDArray

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.oracles.clutteredretrieval2d_medium.approach import (
    ClutteredRetrieval2DOracleApproach,
)
from robocode.oracles.clutteredretrieval2d_medium.obs_helpers import (
    extract_rect,
    extract_robot,
    holding_obstruction_named,
    holding_target_block,
    target_inside_region,
)
from robocode.primitives import build_primitives
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/ClutteredRetrieval2D-o10-v0"
# without waiting for a full solve.
MAX_STEPS = 1000
SEED = 0
DEBUG_LOG_PATH = Path(
    "/home/qianwei/robocode/unit_test_videos/clutteredretrieval2d_medium_debug.log"
)


def _debug_summary(
    state: NDArray,
    approach: ClutteredRetrieval2DOracleApproach,
    step: int,
) -> str:
    """Return a compact per-step debug summary."""
    robot = extract_robot(state)
    target = extract_rect(state, "target_block")
    blocker = approach._remove.blocker_name  # pylint: disable=protected-access
    blocker_pos = "none"
    blocker_holding = False
    if blocker is not None:
        block = extract_rect(state, blocker)
        blocker_pos = f"({block.cx:.2f},{block.cy:.2f})"
        blocker_holding = holding_obstruction_named(state, blocker)
    mode = approach._mode  # pylint: disable=protected-access
    remove_phase = getattr(approach._remove, "_phase", "n/a")  # pylint: disable=protected-access
    staging_index = getattr(approach._remove, "_staging_candidate_index", -1)  # pylint: disable=protected-access
    staging_count = getattr(approach._remove, "_last_staging_count", -1)  # pylint: disable=protected-access
    transport_goal_index = getattr(approach._remove, "_last_transport_goal_index", -1)  # pylint: disable=protected-access
    transport_event = getattr(approach._remove, "_last_transport_event", "n/a")  # pylint: disable=protected-access
    transport_retries = getattr(approach._remove, "_transport_retry_count", -1)  # pylint: disable=protected-access
    grasp_retries = getattr(approach._remove, "_grasp_retry_count", -1)  # pylint: disable=protected-access
    stuck_count = getattr(approach._remove, "_consecutive_stuck_steps", -1)  # pylint: disable=protected-access
    remove_actions = len(approach._remove._actions)  # pylint: disable=protected-access
    acquire_actions = len(approach._acquire._actions)  # pylint: disable=protected-access
    place_actions = len(approach._place._actions)  # pylint: disable=protected-access
    return (
        f"[step {step:03d}] mode={mode} "
        f"remove_phase={remove_phase} "
        f"staging_idx={staging_index}/{staging_count} "
        f"goal_idx={transport_goal_index} "
        f"transport_event={transport_event} "
        f"retries(grasp={grasp_retries},transport={transport_retries}) "
        f"stuck={stuck_count} "
        f"target=({target.cx:.2f},{target.cy:.2f}) "
        f"robot=({robot.x:.2f},{robot.y:.2f},th={robot.theta:.2f},arm={robot.arm_joint:.2f},vac={robot.vacuum:.0f}) "
        f"holding_target={holding_target_block(state)} "
        f"target_in_region={target_inside_region(state)} "
        f"blocker={blocker} "
        f"blocker_pos={blocker_pos} "
        f"holding_blocker={blocker_holding} "
        f"queues(acq={acquire_actions},rm={remove_actions},pl={place_actions}) "
        f"req(acq={approach._acquire.required_blocker},pl={approach._place.required_blocker})"  # pylint: disable=protected-access
    )


def _run_episode(
    episode_env: KinderGeom2DEnv,
    approach: ClutteredRetrieval2DOracleApproach,
    seed: int,
    primitive_env: KinderGeom2DEnv | None = None,
) -> tuple[bool, int]:
    """Run a single episode and return (solved, num_steps)."""
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_LOG_PATH.write_text("", encoding="utf-8")

    def _log(line: str) -> None:
        print(line)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

    state, info = episode_env.reset(seed=seed)
    if primitive_env is not None:
        primitive_env.reset(seed=seed)
        primitive_env.set_state(state)
    approach.reset(state, info)
    _log(_debug_summary(state, approach, 0))

    for step in range(MAX_STEPS):
        action = approach.step()
        state, reward, terminated, truncated, info = episode_env.step(action)
        approach.update(state, float(reward), terminated or truncated, info)
        _log(
            f"  action={action.tolist()} reward={float(reward):.1f} "
            f"terminated={terminated} truncated={truncated}"
        )
        _log(_debug_summary(state, approach, step + 1))
        if terminated or truncated:
            return bool(terminated), step + 1

    return False, MAX_STEPS


@pytest.fixture(name="clutteredretrieval_env")
def _clutteredretrieval_env() -> KinderGeom2DEnv:
    """Create a KinderGeom2DEnv for the ClutteredRetrieval2D-o10 environment."""
    return KinderGeom2DEnv(ENV_ID)


def test_oracle_runs_single_episode(clutteredretrieval_env: KinderGeom2DEnv) -> None:
    """Run one episode and optionally save a video for manual inspection."""
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    episode_env = clutteredretrieval_env
    primitive_env = clutteredretrieval_env
    if MAKE_VIDEOS:
        episode_env = RecordVideo(
            kinder.make(ENV_ID, render_mode=render_mode),
            f"unit_test_videos/clutteredretrieval2d_medium_seed{SEED}",
        )

    primitives = build_primitives(primitive_env, ["check_action_collision", "BiRRT"])
    approach = ClutteredRetrieval2DOracleApproach(
        action_space=episode_env.action_space,
        observation_space=episode_env.observation_space,
        primitives=primitives,
    )
    solved, steps = _run_episode(episode_env, approach, SEED, primitive_env)
    summary = f"seed={SEED}: solved={solved} steps={steps}"
    print(summary)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n")

    if MAKE_VIDEOS:
        episode_env.close()
