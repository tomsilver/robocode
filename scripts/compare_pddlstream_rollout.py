"""Differential test: our PR2 packed environment against stock PDDLStream.

PDDLStream has no ``step()`` -- it plans, then optionally replays a plan
open-loop -- so there is no rollout on that side to compare against directly.
What is shared, and what this script checks, is the *substrate*: the scene, and
the kinematics / collision / placement predicates the dynamics are built from.

Three checks, run in separate processes so each imports its own tree:

1. **Scene.** Build ``packed`` on both sides at the same numpy seed and compare
   body ids, poses, AABBs, and every joint angle.
2. **Rollout.** Roll episodes in our environment (random or oracle actions) and
   record every state visited.
3. **Predicates.** Replay each recorded state into a scene built from stock
   ``examples.pybullet.tamp.problems.packed`` and evaluate the stock library's
   own ``pairwise_collision`` / ``is_placement`` / ``stable_z`` / forward
   kinematics. Compare against ours at the same states.

Check 3 is the load-bearing one: our ``step()`` accepts or rejects a motion on
``pairwise_collision`` and terminates on ``is_placement``, so if the stock tree
agrees on every visited state then our dynamics are stock ss-pybullet physics,
not a private approximation of it.

What this cannot show is that our *transition semantics* match PDDLStream's,
because PDDLStream has none to match. Our action space, kinematic stepping, and
grasp model are additions; only the world they run in is being compared.

Usage::

    python scripts/compare_pddlstream_rollout.py                  # all checks
    python scripts/compare_pddlstream_rollout.py --policy oracle
    python scripts/compare_pddlstream_rollout.py --counts 3 5 --episodes 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# The stock PDDLStream tree is imported inside `_run_stock`, after that tree is put
# on sys.path. It is deliberately not importable at type-check time.
# mypy: disable-error-code=import-not-found
# pylint: disable=import-outside-toplevel,import-error,no-name-in-module


REPO_ROOT = Path(__file__).resolve().parents[1]
STOCK_TREE = REPO_ROOT / "third-party" / "pddlstream"

# Tolerance for comparing two independently computed float trajectories. The two
# processes run the same C++ Bullet routines on the same inputs, so agreement is
# expected to be exact; this only absorbs JSON round-tripping.
TOL = 1e-9


# --------------------------------------------------------------------------
# child: our tree
# --------------------------------------------------------------------------


def _run_ours(
    counts: list[int], episodes: int, steps: int, policy: str, render_dir: str | None
) -> dict:
    """Roll out in our environment and record states plus our own predicates."""
    import numpy as np

    from robocode.environments import pr2_tamp_env as _env_mod
    from robocode.environments.pr2_tamp_env import PR2PackedEnv
    from robocode.environments.ss_pybullet import (
        get_aabb,
        get_joint_positions,
        get_link_pose,
        get_pose,
        is_placement,
        pairwise_collision,
        set_client,
        stable_z,
    )

    def _measure(
        robot: int, base_joints, arm_joints, tool_link, blocks, table: int, plate: int
    ) -> dict:
        """Everything our dynamics read, at the current state.

        The gripper fingers are part of the state, not decoration: closing them is what
        makes the robot touch a held block, so a replay that left them open would
        disagree about that contact for reasons of its own.
        """
        from robocode.environments.ss_pybullet import get_gripper_joints

        gripper_joints = list(get_gripper_joints(robot, "left"))
        return {
            "gripper": [float(x) for x in get_joint_positions(robot, gripper_joints)],
            "base": [float(x) for x in get_joint_positions(robot, base_joints)],
            "arm": [float(x) for x in get_joint_positions(robot, arm_joints)],
            "block_poses": {
                str(b): [
                    list(map(float, get_pose(b)[0])),
                    list(map(float, get_pose(b)[1])),
                ]
                for b in blocks
            },
            "tool": [
                list(map(float, get_link_pose(robot, tool_link)[0])),
                list(map(float, get_link_pose(robot, tool_link)[1])),
            ],
            "collide_table": bool(pairwise_collision(robot, table)),
            "collide_plate": bool(pairwise_collision(robot, plate)),
            "collide_block": {
                str(b): bool(pairwise_collision(robot, b)) for b in blocks
            },
            "on_plate": {str(b): bool(is_placement(b, plate)) for b in blocks},
            "on_table": {str(b): bool(is_placement(b, table)) for b in blocks},
            "aabb": {
                str(b): [
                    list(map(float, get_aabb(b)[0])),
                    list(map(float, get_aabb(b)[1])),
                ]
                for b in blocks
            },
            "stable_z_plate": {str(b): float(stable_z(b, plate)) for b in blocks},
        }

    def _scene_snapshot(handles: dict) -> dict:
        """Measure a freshly built scene, addressing it by its handles."""
        from robocode.environments.ss_pybullet import (
            PR2_TOOL_FRAMES,
            get_arm_joints,
            get_group_joints,
            link_from_name,
        )

        robot = handles["robot"]
        return _measure(
            robot,
            list(get_group_joints(robot, "base")),
            list(get_arm_joints(robot, "left")),
            link_from_name(robot, PR2_TOOL_FRAMES["left"]),
            handles["blocks"],
            handles["table"],
            handles["plate"],
        )

    def snapshot(env: PR2PackedEnv) -> dict:
        """Measure the live environment at its current state."""
        set_client(env._client)  # pylint: disable=protected-access
        return _measure(
            env.robot,
            env.base_joints,
            env.arm_joints,
            env.tool_link,
            env.blocks,
            env.table,
            env.plate,
        )

    out: dict[str, Any] = {
        "scene": {},
        "episodes": [],
        # The stock side renders with exactly this camera, so any visual difference
        # is the world differing, not the viewpoint. Read from the environment
        # module rather than restated here, so the two can never drift apart.
        # pylint: disable=protected-access
        "camera": {
            "eye": list(_env_mod._CAMERA_EYE),
            "target": list(_env_mod._CAMERA_TARGET),
            "fov": _env_mod._CAMERA_FOV_DEGREES,
            "near": _env_mod._CAMERA_NEAR,
            "far": _env_mod._CAMERA_FAR,
            "width": _env_mod._RENDER_WIDTH,
            "height": _env_mod._RENDER_HEIGHT,
        },
    }

    # Scene check: build the vendored scene directly at a fixed numpy seed, which is
    # exactly how stock `packed` is driven. It cannot go through `reset()`, because
    # reset re-seeds numpy from the episode's generator to make instances
    # reproducible per eval seed, which would overwrite the seed set here.
    for count in counts:
        from robocode.environments.pr2_tamp_scenes import build_packed_scene
        from robocode.environments.ss_pybullet import (
            HideOutput,
            connect,
            disconnect,
        )

        client = connect(use_gui=False)
        set_client(client)
        try:
            np.random.seed(0)
            with HideOutput():
                _problem, handles = build_packed_scene(count)
            out["scene"][str(count)] = _scene_snapshot(handles)
        finally:
            set_client(client)
            disconnect()

    for count in counts:
        env = PR2PackedEnv(num_blocks=count)
        try:
            for ep in range(episodes):
                seed = 1000 + ep
                env.action_space.seed(seed)
                env.reset(seed=seed)
                approach = None
                if policy == "oracle":
                    from robocode.oracles.pr2packed.approach import (
                        PR2PackedOracleApproach,
                    )

                    approach = PR2PackedOracleApproach(
                        env.action_space, env.observation_space, seed=seed, env=env
                    )
                    approach.reset(env.get_state(), {})

                frames = [snapshot(env)]
                actions = []
                shots = []
                if render_dir:
                    shots.append(
                        _save(env.render(), render_dir, "ours", len(out["episodes"]), 0)
                    )
                for _ in range(steps):
                    if approach is not None:
                        action = approach.step()
                    else:
                        action = env.action_space.sample()
                    obs, reward, term, trunc, info = env.step(action)
                    if approach is not None:
                        approach.update(obs, float(reward), term or trunc, info)
                    actions.append([float(a) for a in action])
                    frames.append(snapshot(env))
                    if render_dir:
                        shots.append(
                            _save(
                                env.render(),
                                render_dir,
                                "ours",
                                len(out["episodes"]),
                                len(frames) - 1,
                            )
                        )
                    if term or trunc:
                        break
                out["episodes"].append(
                    {
                        "count": count,
                        "seed": seed,
                        "actions": actions,
                        "frames": frames,
                        "shots": shots,
                    }
                )
        finally:
            env.close()
    return out


# --------------------------------------------------------------------------
# child: stock tree
# --------------------------------------------------------------------------


def _run_stock(payload: dict) -> dict:
    """Replay our recorded states in a stock scene and evaluate stock predicates."""
    sys.path.insert(0, str(STOCK_TREE))
    sys.path.insert(0, str(STOCK_TREE / "examples" / "pybullet" / "utils" / "motion"))
    import numpy as np
    from examples.pybullet.tamp.problems import packed
    from examples.pybullet.utils.pybullet_tools.pr2_utils import (
        PR2_TOOL_FRAMES,
        get_arm_joints,
        get_gripper_joints,
        get_group_joints,
    )
    from examples.pybullet.utils.pybullet_tools.utils import (
        HideOutput,
        connect,
        disconnect,
        get_aabb,
        get_bodies,
        get_joint_name,
        get_joint_positions,
        get_link_pose,
        get_movable_joints,
        get_pose,
        is_placement,
        link_from_name,
        pairwise_collision,
        set_client,
        set_joint_positions,
        set_pose,
        stable_z,
    )

    results: dict[str, Any] = {"scene": {}, "episodes": []}

    def build(count: int) -> dict:
        """Build the stock scene in the current client and return its handles."""
        np.random.seed(0)
        with HideOutput():
            problem = packed(num=count)
        robot = problem.robot
        return {
            "robot": robot,
            "blocks": list(problem.movable),
            # `packed` builds table then plate, so surfaces == [table, plate].
            "table": problem.surfaces[0],
            "plate": problem.surfaces[1],
            "base_joints": list(get_group_joints(robot, "base")),
            "arm_joints": list(get_arm_joints(robot, "left")),
            "gripper_joints": list(get_gripper_joints(robot, "left")),
            "tool_link": link_from_name(robot, PR2_TOOL_FRAMES["left"]),
        }

    def evaluate(scene: dict) -> dict:
        robot, blocks = scene["robot"], scene["blocks"]
        return {
            "gripper": [
                float(x) for x in get_joint_positions(robot, scene["gripper_joints"])
            ],
            "base": [
                float(x) for x in get_joint_positions(robot, scene["base_joints"])
            ],
            "arm": [float(x) for x in get_joint_positions(robot, scene["arm_joints"])],
            "block_poses": {
                str(b): [
                    list(map(float, get_pose(b)[0])),
                    list(map(float, get_pose(b)[1])),
                ]
                for b in blocks
            },
            "tool": [
                list(map(float, get_link_pose(robot, scene["tool_link"])[0])),
                list(map(float, get_link_pose(robot, scene["tool_link"])[1])),
            ],
            "collide_table": bool(pairwise_collision(robot, scene["table"])),
            "collide_plate": bool(pairwise_collision(robot, scene["plate"])),
            "collide_block": {
                str(b): bool(pairwise_collision(robot, b)) for b in blocks
            },
            "on_plate": {str(b): bool(is_placement(b, scene["plate"])) for b in blocks},
            "on_table": {str(b): bool(is_placement(b, scene["table"])) for b in blocks},
            "aabb": {
                str(b): [
                    list(map(float, get_aabb(b)[0])),
                    list(map(float, get_aabb(b)[1])),
                ]
                for b in blocks
            },
            "stable_z_plate": {
                str(b): float(stable_z(b, scene["plate"])) for b in blocks
            },
        }

    def restore(scene: dict, frame: dict) -> None:
        """Put the stock scene into the state our env was in."""
        set_joint_positions(scene["robot"], scene["base_joints"], frame["base"])
        set_joint_positions(scene["robot"], scene["arm_joints"], frame["arm"])
        set_joint_positions(scene["robot"], scene["gripper_joints"], frame["gripper"])
        for body, (point, quat) in frame["block_poses"].items():
            set_pose(int(body), (tuple(point), tuple(quat)))

    # One fresh client per object count. Stock `sample_placements` loops with an
    # unbounded `while True`, so building a second scene into a world that already
    # holds one never terminates: the table is already covered.
    counts = sorted(
        {int(c) for c in payload["scene"]}
        | {int(e["count"]) for e in payload["episodes"]}
    )
    for count in counts:
        client = connect(use_gui=False)
        set_client(client)
        try:
            scene = build(count)
            if str(count) in payload["scene"]:
                snap = evaluate(scene)
                snap["bodies"] = get_bodies()
                snap["all_joints"] = {
                    str(b): [
                        [get_joint_name(b, j), round(float(v), 12)]
                        for j, v in zip(
                            get_movable_joints(b),
                            get_joint_positions(b, get_movable_joints(b)),
                        )
                    ]
                    for b in get_bodies()
                }
                results["scene"][str(count)] = snap
            for index, episode in enumerate(payload["episodes"]):
                if episode["count"] != count:
                    continue
                frames, shots = [], []
                for step, frame in enumerate(episode["frames"]):
                    restore(scene, frame)
                    frames.append(evaluate(scene))
                    if payload.get("render_dir"):
                        shots.append(
                            _save(
                                _shoot(client, payload["camera"]),
                                payload["render_dir"],
                                "stock",
                                index,
                                step,
                            )
                        )
                results["episodes"].append(
                    {
                        "count": count,
                        "seed": episode["seed"],
                        "frames": frames,
                        "shots": shots,
                    }
                )
        finally:
            set_client(client)
            disconnect()

    return results


def _save(frame: Any, render_dir: str, side: str, episode: int, step: int) -> str:
    """Write one RGB frame and return its path."""
    import imageio.v3 as iio
    import numpy as np

    out = Path(render_dir) / side
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ep{episode:02d}_f{step:04d}.png"
    iio.imwrite(path, np.asarray(frame, dtype="uint8"))
    return str(path)


def _shoot(client: int, camera: dict) -> Any:
    """Render the stock scene through our environment's camera."""
    import numpy as np
    import pybullet as pb

    view = pb.computeViewMatrix(
        camera["eye"], camera["target"], [0, 0, 1], physicsClientId=client
    )
    projection = pb.computeProjectionMatrixFOV(
        camera["fov"],
        camera["width"] / camera["height"],
        camera["near"],
        camera["far"],
        physicsClientId=client,
    )
    _, _, rgba, _, _ = pb.getCameraImage(
        camera["width"],
        camera["height"],
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=pb.ER_TINY_RENDERER,
        shadow=False,
        physicsClientId=client,
    )
    return np.asarray(rgba, dtype="uint8").reshape(
        camera["height"], camera["width"], 4
    )[:, :, :3]


def _contact_sheet(ours: dict, stock: dict, render_dir: str) -> list[str]:
    """Build one side-by-side GIF per episode, with a per-frame pixel difference."""
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image, ImageDraw

    made = []
    for index, (a, b) in enumerate(zip(ours["episodes"], stock["episodes"])):
        if not a.get("shots") or not b.get("shots"):
            continue
        panels, worst = [], 0.0
        for left_path, right_path in zip(a["shots"], b["shots"]):
            left = np.asarray(Image.open(left_path).convert("RGB"))
            right = np.asarray(Image.open(right_path).convert("RGB"))
            delta = np.abs(left.astype(int) - right.astype(int))
            worst = max(worst, float(delta.max()))
            gap = 8
            sheet = Image.new(
                "RGB", (left.shape[1] * 2 + gap, left.shape[0] + 26), "white"
            )
            sheet.paste(Image.fromarray(left), (0, 26))
            sheet.paste(Image.fromarray(right), (left.shape[1] + gap, 26))
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), "ours (PR2PackedEnv)", fill=(0, 0, 0))
            draw.text((left.shape[1] + gap + 8, 8), "stock PDDLStream", fill=(0, 0, 0))
            draw.text(
                (left.shape[1] - 120, 8),
                f"max pixel diff {int(delta.max())}",
                fill=(180, 0, 0) if delta.max() else (0, 130, 0),
            )
            panels.append(np.asarray(sheet))
        path = Path(render_dir) / f"compare_ep{index:02d}_count{a['count']}.gif"
        iio.imwrite(path, panels, extension=".gif", loop=0, fps=8)
        made.append(f"{path}  (frames={len(panels)}, max pixel diff={int(worst)})")
    return made


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def _diff(ours: Any, stock: Any, path: str, out: list[str]) -> None:
    """Walk two nested structures, recording every disagreement."""
    if isinstance(ours, dict):
        if set(ours) != set(stock):
            out.append(f"{path}: key mismatch {sorted(set(ours) ^ set(stock))}")
            return
        for key in ours:
            _diff(ours[key], stock[key], f"{path}.{key}", out)
    elif isinstance(ours, list):
        if len(ours) != len(stock):
            out.append(f"{path}: length {len(ours)} vs {len(stock)}")
            return
        for i, (a, b) in enumerate(zip(ours, stock)):
            _diff(a, b, f"{path}[{i}]", out)
    elif isinstance(ours, bool) or isinstance(stock, bool):
        if bool(ours) != bool(stock):
            out.append(f"{path}: {ours} vs {stock}")
    elif isinstance(ours, (int, float)):
        if abs(float(ours) - float(stock)) > TOL:
            out.append(f"{path}: {ours!r} vs {stock!r} (|d|={abs(ours - stock):.3g})")
    elif ours != stock:
        out.append(f"{path}: {ours!r} vs {stock!r}")


def _child(mode: str, payload_path: str | None, out_path: str, args: Any) -> int:
    if mode == "ours":
        result = _run_ours(
            args.counts, args.episodes, args.steps, args.policy, args.render
        )
    else:
        assert payload_path is not None
        result = _run_stock(json.loads(Path(payload_path).read_text(encoding="utf-8")))
    Path(out_path).write_text(json.dumps(result), encoding="utf-8")
    return 0


def main() -> int:
    """Run the three checks and report any disagreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--policy", choices=["random", "oracle"], default="random")
    parser.add_argument(
        "--mode", choices=["compare", "ours", "stock"], default="compare"
    )
    parser.add_argument("--render", help="directory for side-by-side frames and GIFs")
    parser.add_argument("--payload")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.mode != "compare":
        return _child(args.mode, args.payload, args.out, args)

    if not (STOCK_TREE / "examples" / "pybullet" / "tamp" / "problems.py").exists():
        print(f"stock PDDLStream tree not found at {STOCK_TREE}", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="pddlstream-compare-"))
    ours_path, stock_path = tmp / "ours.json", tmp / "stock.json"
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--counts",
        *map(str, args.counts),
        "--episodes",
        str(args.episodes),
        "--steps",
        str(args.steps),
        "--policy",
        args.policy,
    ]
    if args.render:
        Path(args.render).mkdir(parents=True, exist_ok=True)
        base += ["--render", str(Path(args.render).resolve())]

    print(f"rolling out in our environment ({args.policy} policy)...")
    subprocess.run(base + ["--mode", "ours", "--out", str(ours_path)], check=True)
    if args.render:
        payload = json.loads(ours_path.read_text(encoding="utf-8"))
        payload["render_dir"] = str(Path(args.render).resolve())
        ours_path.write_text(json.dumps(payload), encoding="utf-8")
    print("replaying those states in the stock PDDLStream tree...")
    subprocess.run(
        base
        + ["--mode", "stock", "--payload", str(ours_path), "--out", str(stock_path)],
        check=True,
        cwd=STOCK_TREE,
    )

    ours = json.loads(ours_path.read_text(encoding="utf-8"))
    if args.render:
        ours["render_dir"] = str(Path(args.render).resolve())
        ours_path.write_text(json.dumps(ours), encoding="utf-8")
    stock = json.loads(stock_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    for count_str, snap in ours["scene"].items():
        _diff(
            snap,
            {k: v for k, v in stock["scene"][count_str].items() if k in snap},
            f"scene[{count_str}]",
            failures,
        )
    states = 0
    for ours_ep, stock_ep in zip(ours["episodes"], stock["episodes"]):
        tag = f"episode(count={ours_ep['count']},seed={ours_ep['seed']})"
        for i, (a, b) in enumerate(zip(ours_ep["frames"], stock_ep["frames"])):
            _diff(a, b, f"{tag}.frame[{i}]", failures)
            states += 1

    print()
    print(f"scenes compared      : {len(ours['scene'])}")
    print(f"episodes rolled out  : {len(ours['episodes'])}")
    print(f"states cross-checked : {states}")
    print(f"values per state     : ~{2 * len(ours['episodes'][0]['frames'][0])}+")
    if failures:
        print(f"\nMISMATCHES ({len(failures)}):")
        for line in failures[:40]:
            print("  " + line)
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        return 1
    if args.render:
        print("\nside-by-side renders:")
        for line in _contact_sheet(ours, stock, str(Path(args.render).resolve())):
            print("  " + line)
    print("\nOK: stock PDDLStream agrees with our environment at every state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
