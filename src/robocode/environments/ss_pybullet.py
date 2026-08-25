"""Import shim for the ``ss-pybullet`` submodule.

``ss-pybullet`` and its nested ``motion-planners`` submodule ship no packaging
metadata (no ``setup.py``, no ``pyproject.toml``), so neither can be declared as a
``uv`` path dependency the way ``kindergarden`` is. They are plain source trees
that have to be imported off ``sys.path``, and ``pybullet_tools.utils`` imports
``motion_planners`` at module scope, so both directories must be on the path
before the first ``pybullet_tools`` import.

This module does that once and re-exports the symbols the PR2 environments use,
so no other robocode module manipulates ``sys.path``. Import it instead of
importing ``pybullet_tools`` directly; importing ``pybullet_tools`` through two
different paths loads two copies of the module, and its ``CLIENT``/``CLIENTS``
globals then disagree about which physics client is live.
"""

# The sys.path bootstrap below must run before the pybullet_tools imports, so those
# imports cannot sit at the top of the module. Moving the bootstrap into a separate
# module imported first does not help: isort sorts `pybullet_tools` ahead of
# `robocode`, so it hoists the real imports above the bootstrap again, and `isort:
# skip` only pins the marked line rather than the ones around it.
# pylint: disable=wrong-import-position

from __future__ import annotations

import sys
from pathlib import Path

_SS_PYBULLET_ROOT = Path(__file__).resolve().parents[3] / "third-party" / "ss-pybullet"
_MOTION_PLANNERS_ROOT = _SS_PYBULLET_ROOT / "motion"


def _ensure_on_path() -> None:
    """Put ss-pybullet and motion-planners on ``sys.path`` (idempotent)."""
    if not (_SS_PYBULLET_ROOT / "pybullet_tools").is_dir():
        raise ImportError(
            f"ss-pybullet is not checked out at {_SS_PYBULLET_ROOT}. Run "
            "`git submodule update --init --recursive` (install.sh does this)."
        )
    if not (_MOTION_PLANNERS_ROOT / "motion_planners").is_dir():
        raise ImportError(
            f"ss-pybullet's nested motion-planners submodule is not checked out at "
            f"{_MOTION_PLANNERS_ROOT}. Run `git submodule update --init --recursive`."
        )
    for root in (_SS_PYBULLET_ROOT, _MOTION_PLANNERS_ROOT):
        entry = str(root)
        if entry not in sys.path:
            sys.path.insert(0, entry)


_ensure_on_path()

from pybullet_tools.pr2_problems import (  # noqa: E402
    Problem,
    create_pr2,
    create_table,
)
from pybullet_tools.pr2_utils import (  # noqa: E402
    PR2_TOOL_FRAMES,
    REST_LEFT_ARM,
    arm_conf,
    close_arm,
    get_arm_joints,
    get_carry_conf,
    get_gripper_joints,
    get_group_joints,
    get_other_arm,
    get_top_grasps,
    open_arm,
    set_arm_conf,
    set_group_conf,
)
from pybullet_tools.utils import (  # noqa: E402
    BLUE,
    GREEN,
    Attachment,
    Euler,
    HideOutput,
    Point,
    Pose,
    WorldSaver,
    aabb2d_from_aabb,
    aabb_contains_point,
    add_data_path,
    connect,
    create_attachment,
    create_box,
    disconnect,
    get_aabb,
    get_bodies,
    get_client,
    get_joint_limits,
    get_joint_positions,
    get_link_pose,
    get_pose,
    invert,
    is_circular,
    is_placement,
    link_from_name,
    load_pybullet,
    multiply,
    pairwise_collision,
    plan_joint_motion,
    sample_placement,
    set_client,
    set_joint_positions,
    set_point,
    set_pose,
    stable_z,
    sub_inverse_kinematics,
)

__all__ = [
    "get_client",
    "multiply",
    "invert",
    "sub_inverse_kinematics",
    "plan_joint_motion",
    "is_circular",
    "get_top_grasps",
    "WorldSaver",
    "Pose",
    "Euler",
    "Attachment",
    "BLUE",
    "GREEN",
    "HideOutput",
    "PR2_TOOL_FRAMES",
    "Point",
    "Problem",
    "REST_LEFT_ARM",
    "aabb2d_from_aabb",
    "aabb_contains_point",
    "add_data_path",
    "arm_conf",
    "close_arm",
    "connect",
    "create_attachment",
    "create_box",
    "create_pr2",
    "create_table",
    "disconnect",
    "get_aabb",
    "get_arm_joints",
    "get_bodies",
    "get_carry_conf",
    "get_gripper_joints",
    "get_group_joints",
    "get_joint_limits",
    "get_joint_positions",
    "get_link_pose",
    "get_other_arm",
    "get_pose",
    "is_placement",
    "link_from_name",
    "load_pybullet",
    "open_arm",
    "pairwise_collision",
    "sample_placement",
    "set_arm_conf",
    "set_client",
    "set_group_conf",
    "set_joint_positions",
    "set_point",
    "set_pose",
    "stable_z",
]
