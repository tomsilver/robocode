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
_PANEL_SIZE = (480, 408)
_IMAGE_SIZE = (480, 360)


def _create_environment(config_name: str) -> VariableObjectCountEnv:
    with initialize_config_dir(config_dir=str(_CONF_DIR), version_base=None):
        config = compose(config_name="config", overrides=[f"environment={config_name}"])
    return cast(VariableObjectCountEnv, instantiate(config.environment))


def _panel(frame: NDArray[Any], label: str) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    image.thumbnail(_IMAGE_SIZE, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", _PANEL_SIZE, "white")
    x = (_PANEL_SIZE[0] - image.width) // 2
    y = 48 + (_IMAGE_SIZE[1] - image.height) // 2
    panel.paste(image, (x, y))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    draw.text((12, 12), label, fill="black", font=font)
    return panel


def render_gif(output: Path, num_frames: int) -> None:
    """Render synchronized initial-state samples from all generalized families."""
    environments = [
        (label, _create_environment(config_name))
        for label, config_name in _ENVIRONMENTS
    ]
    frames: list[Image.Image] = []
    try:
        for seed in range(num_frames):
            panels = []
            for label, env in environments:
                _, info = env.reset(seed=seed)
                frame = cast(NDArray[Any], env.render())
                panels.append(
                    _panel(
                        frame, f"{label} | count={info['object_count']} | seed={seed}"
                    )
                )
            montage = Image.new(
                "RGB", (_PANEL_SIZE[0] * len(panels), _PANEL_SIZE[1]), "white"
            )
            for index, panel in enumerate(panels):
                montage.paste(panel, (index * _PANEL_SIZE[0], 0))
            frames.append(montage)
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
    parser.add_argument("--num-frames", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    render_gif(args.output, args.num_frames)
