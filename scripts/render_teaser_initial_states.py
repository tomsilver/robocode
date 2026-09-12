"""Render curated frozen reset states as camera-sweep MP4s and clean PNGs.

Run with MUJOCO_GL=egl .venv/bin/python scripts/render_teaser_initial_states.py.
Camera coordinates are world-space target, distance, azimuth and elevation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import pybullet as p
from hydra.utils import instantiate
from kinder.envs.dynamic3d.envs import TidyBot3DEnv
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/teaser_initial_states"
# name, config (or task JSON stem), seed, pinned count, camera
BLOCKED = ([3.6, 0.0, 0.55], 4.65, -55.0, 35.0)
BLOCKED_ALT = ([3.6, 0.0, 0.55], 4.65, 65.0, 35.0)
PACKED = ([-0.4, 0.0, 0.8], 4.05, -45.0, 35.0)
ROVERS = ([0.0, 0.0, 0.0], 9.5, -75.0, 52.0)
SORT = ([0.43, 0.0, 0.4], 3.35, 135.0, 35.0)
JOBS = [
    ("pr2_blocked_seed42", "pr2blocked_easy", 42, None, BLOCKED),
    ("pr2_blocked_seed24", "pr2blocked_easy", 24, None, BLOCKED_ALT),
    ("pr2_blocked_variable_spares0_seed42", "pr2blocked_generalized", 42, 0, BLOCKED),
    (
        "pr2_blocked_variable_spares4_seed24",
        "pr2blocked_generalized",
        24,
        4,
        BLOCKED_ALT,
    ),
    ("pr2_packed_variable_blocks3_seed42", "pr2packed_generalized", 42, 3, PACKED),
    ("pr2_packed_variable_blocks5_seed24", "pr2packed_generalized", 24, 5, PACKED),
    ("rovers_seed42", "rovers_hard", 42, None, ROVERS),
    ("rovers_seed24", "rovers_hard", 24, None, ROVERS),
    (
        "sort_bins4_seed42",
        "SortClutteredBlocks3D-o4-sort_the_cluttered_blocks_into_bins",
        42,
        None,
        SORT,
    ),
    (
        "sort_bins20_seed24",
        "SortClutteredBlocks3D-o20-sort_the_cluttered_blocks_into_bins",
        24,
        None,
        SORT,
    ),
    (
        "sort_bowls20_seed42",
        "SortClutteredBlocks3D-o20-sort_the_cluttered_blocks_into_bowls",
        42,
        None,
        SORT,
    ),
    (
        "sort_cupboard12_seed24",
        "SortClutteredBlocks3D-o12-sort_the_blocks_into_the_cupboard",
        24,
        None,
        ([1.0, 0.0, 0.3], 5.0, 150.0, 45.0),
    ),
]


def render_frame(backend, renderer, camera, width, height, offset=0.0):
    """Render either simulator without advancing its state."""
    target, distance, azimuth, elevation = camera
    if renderer is not None:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = target
        cam.distance = distance
        cam.azimuth = azimuth + offset + 180
        cam.elevation = -elevation
        renderer.update_scene(backend.data.mj_data, camera=cam)
        return renderer.render()
    azimuth, elevation = np.deg2rad([azimuth + offset, elevation])
    eye = np.asarray(target) + distance * np.array(
        [
            np.cos(azimuth) * np.cos(elevation),
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ]
    )
    rgba = p.getCameraImage(
        width,
        height,
        viewMatrix=p.computeViewMatrix(eye.tolist(), target, [0, 0, 1]),
        projectionMatrix=p.computeProjectionMatrixFOV(40, width / height, 0.02, 80),
        renderer=p.ER_TINY_RENDERER,
        shadow=False,
        physicsClientId=backend._client,
    )[2]
    return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]


def main():
    """Export the requested views and a contact sheet with reproducibility data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", nargs="+", choices=[job[0] for job in JOBS])
    parser.add_argument("--stills-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    width, height, fps, frames = 1440, 1080, 24, 96
    for name, config, seed, count, camera in JOBS:
        if args.jobs and name not in args.jobs:
            continue
        print(f"Starting {name}", flush=True)
        is_sort = name.startswith("sort_")
        if is_sort:
            task = (
                ROOT
                / "third-party/kindergarden/src/kinder/envs/dynamic3d/tasks/SortClutteredBlocks3D"
                / f"{config}.json"
            )
            env = TidyBot3DEnv(
                task_config_path=str(task),
                scene_bg=False,
                scene_render_camera="task_view",
                render_mode="rgb_array",
            )
        else:
            env = instantiate(
                OmegaConf.load(ROOT / "experiments/conf/environment" / f"{config}.yaml")
            )
        renderer = None
        try:
            env.reset(
                seed=seed, options=None if count is None else {"object_count": count}
            )
            backend = (
                env._object_centric_env._robot_env.sim
                if is_sort
                else getattr(env, "current_backend", env)
            )
            if is_sort:
                before = np.concatenate(
                    [backend.data.mj_data.qpos.copy(), backend.data.mj_data.qvel.copy()]
                )
                model = backend.model.mj_model
                model.vis.global_.offwidth = width * 2
                model.vis.global_.offheight = height * 2
                model.vis.global_.fovy = 40
                renderer = mujoco.Renderer(model, height * 2, width * 2)
            else:
                with backend.client():
                    before = backend._get_obs().copy()
                if name.startswith("pr2_"):
                    p.changeVisualShape(
                        0,
                        -1,
                        textureUniqueId=-1,
                        rgbaColor=[0.94, 0.94, 0.94, 1],
                        physicsClientId=backend._client,
                    )
            still = render_frame(backend, renderer, camera, width * 2, height * 2)
            Image.fromarray(still).save(OUTPUT / f"{name}.png")
            # Both original table locations are shown in an additional overview.
            if "blocked" in name:
                overview = ([0.0, 0.0, 0.6], 13.0, -85.0, 45.0)
                Image.fromarray(
                    render_frame(backend, renderer, overview, width * 2, height * 2)
                ).save(OUTPUT / f"{name}_overview.png")
            if renderer is not None:
                renderer.close()
                renderer = mujoco.Renderer(model, height, width)
            if not args.stills_only:
                with imageio.get_writer(
                    OUTPUT / f"{name}.mp4",
                    fps=fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=1,
                    ffmpeg_params=["-crf", "17", "-movflags", "+faststart"],
                ) as writer:
                    for index in range(frames):
                        # One gentle oscillation, with identical endpoints for looping.
                        offset = 9 * np.sin(2 * np.pi * index / (frames - 1))
                        writer.append_data(
                            render_frame(
                                backend, renderer, camera, width, height, offset
                            )
                        )
                        if index % 24 == 0:
                            print(f"  {name}: {index}/{frames} frames", flush=True)
            if is_sort:
                after = np.concatenate(
                    [backend.data.mj_data.qpos, backend.data.mj_data.qvel]
                )
            else:
                with backend.client():
                    after = backend._get_obs()
            np.testing.assert_array_equal(before, after)
            manifest[name] = dict(
                config=config,
                seed=seed,
                pinned_count=count,
                camera=camera,
                vertical_fov=40,
                video_size=[width, height],
                still_size=[width * 2, height * 2],
                fps=fps,
                frames=frames,
                sweep_degrees=9,
                frozen_state_verified=True,
                png=f"{name}.png",
                mp4=f"{name}.mp4" if not args.stills_only else None,
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        finally:
            if renderer is not None:
                renderer.close()
            env.close()
        print(f"Finished {name}", flush=True)
    entries = [
        (name, record)
        for name, record in manifest.items()
        if (OUTPUT / record["png"]).exists()
    ]
    sheet = Image.new("RGB", (1600, 330 * ((len(entries) + 3) // 4)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    for index, (name, record) in enumerate(entries):
        x, y = index % 4 * 400, index // 4 * 330
        with Image.open(OUTPUT / record["png"]) as still:
            sheet.paste(still.resize((400, 300), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 6, y + 305), name, fill="black", font=font)
    sheet.save(OUTPUT / "contact_sheet.jpg", quality=95)
    cards = []
    for name, record in entries:
        overview_link = (
            f' · <a href="{name}_overview.png">Both tables</a>'
            if "blocked" in name
            else ""
        )
        media = (
            f'<video controls loop muted playsinline preload="none" '
            f'poster="{record["png"]}" src="{record["mp4"]}"></video>'
            if record["mp4"]
            else f'<img src="{record["png"]}">'
        )
        cards.append(
            f'<article>{media}<p>{name.replace("_", " ")}<br>'
            f'<a href="{record["png"]}">Full-resolution PNG</a>{overview_link}</p></article>'
        )
    (OUTPUT / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Teaser initial states</title><style>"
        "body{font:16px system-ui;margin:32px;background:#fafafa;color:#222}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:24px}"
        "article{background:white;border:1px solid #ddd;border-radius:8px;overflow:hidden}"
        "video,img{width:100%;display:block}article p{padding:0 16px;line-height:1.8}"
        "a{color:#2456a6}</style><h1>Teaser initial states</h1>"
        "<p>Frozen reset states · 4-second camera sweeps · 1440 × 1080 video · "
        "2880 × 2160 stills</p><p>PR2 floor texture replaced with neutral gray. "
        "Object colors, geometry and poses are preserved. Both-table views keep "
        "the original nine-metre separation. ss_pybullet.py is a shared import "
        "shim and has no separate scene.</p><main>"
        + "".join(cards)
        + "</main></html>\n"
    )


if __name__ == "__main__":
    main()
