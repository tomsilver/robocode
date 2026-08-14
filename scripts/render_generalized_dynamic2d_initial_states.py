"""Render a GIF of generalized Dynamic2D initial-state samples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from robocode.environments.variable_object_count_env import VariableObjectCountEnv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONF_DIR = _REPO_ROOT / "experiments" / "conf"
_ENVIRONMENTS = (
    ("Dynamic Obstruction2D", "dynobstruction2d_generalized"),
    ("Dynamic PushPullHook2D", "dynpushpullhook2d_generalized"),
    ("Dynamic ScoopPour2D", "dynscooppour2d_generalized"),
)
_PANEL_SIZE = (480, 424)
_IMAGE_SIZE = (480, 360)


def _create_environment(config_name: str) -> VariableObjectCountEnv:
    with initialize_config_dir(config_dir=str(_CONF_DIR), version_base=None):
        config = compose(config_name="config", overrides=[f"environment={config_name}"])
    return cast(VariableObjectCountEnv, instantiate(config.environment))


def _panel(
    frame: NDArray[Any], phase: str, label: str, count: int, seed: int
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    image.thumbnail(_IMAGE_SIZE, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", _PANEL_SIZE, "white")
    x = (_PANEL_SIZE[0] - image.width) // 2
    y = 64 + (_IMAGE_SIZE[1] - image.height) // 2
    panel.paste(image, (x, y))
    draw = ImageDraw.Draw(panel)
    header_color = "#dceefa" if phase == "DESIGN" else "#ffe4bc"
    draw.rectangle((0, 0, _PANEL_SIZE[0], 64), fill=header_color)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    draw.multiline_text(
        (12, 7),
        f"{phase} | {label}\ncount={count} | seed={seed}",
        fill="black",
        font=font,
        spacing=2,
    )
    return panel


def _render_montage(
    environments: list[tuple[str, VariableObjectCountEnv]],
    phase: str,
    seed: int,
    counts: list[int] | None = None,
) -> Image.Image:
    panels = []
    for index, (label, env) in enumerate(environments):
        options = None if counts is None else {"object_count": counts[index]}
        _, info = env.reset(seed=seed, options=options)
        frame = cast(NDArray[Any], env.render())
        panels.append(_panel(frame, phase, label, info["object_count"], seed))
    montage = Image.new("RGB", (_PANEL_SIZE[0] * len(panels), _PANEL_SIZE[1]), "white")
    for index, panel in enumerate(panels):
        montage.paste(panel, (index * _PANEL_SIZE[0], 0))
    return montage


def render_gif(output: Path, num_design_frames: int) -> None:
    """Render synchronized initial-state samples from all generalized families."""
    environments = [
        (label, _create_environment(config_name))
        for label, config_name in _ENVIRONMENTS
    ]
    frames: list[Image.Image] = []
    try:
        for seed in range(num_design_frames):
            frames.append(_render_montage(environments, "DESIGN", seed))

        num_eval_frames = max(len(env.eval_counts) for _, env in environments)
        for index in range(num_eval_frames):
            counts = [
                env.eval_counts[index % len(env.eval_counts)] for _, env in environments
            ]
            frames.append(
                _render_montage(
                    environments,
                    "EVAL",
                    seed=num_design_frames + index,
                    counts=counts,
                )
            )
    finally:
        for _, env in environments:
            env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
        optimize=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-design-frames", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    render_gif(args.output, args.num_design_frames)
