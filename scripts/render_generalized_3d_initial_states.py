"""Render seeded initial-state GIFs for generalized 3D environments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import instantiate
from numpy.typing import NDArray
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from robocode.utils.episode import save_video

_CONFIG_DIR = Path(__file__).parents[1] / "experiments" / "conf" / "environment"
_DEFAULT_CONFIGS = (
    "table3d_generalized",
    "transport3d_generalized",
    "shelf3d_generalized",
    "obstruction3d_generalized",
    "packing3d_generalized",
    "constrainedcupboard3d_generalized",
)


def _annotate(
    frame: NDArray[np.uint8], env_name: str, seed: int, count: int
) -> NDArray[np.uint8]:
    rgb = frame[..., :3]
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    label = f"{env_name} | seed={seed:03d} | object_count={count}"
    _, _, _, text_height = draw.textbbox((0, 0), label)
    draw.rectangle((0, 0, image.width, text_height + 8), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return np.asarray(image)


def _render_config(
    env_name: str,
    output_dir: Path,
    num_seeds: int,
    fps: int,
    render_dpi: int,
) -> dict[str, Any]:
    config = OmegaConf.load(_CONFIG_DIR / f"{env_name}.yaml")
    config.render_dpi = render_dpi
    env = instantiate(config)
    frames: list[NDArray[np.uint8]] = []
    counts: Counter[int] = Counter()
    try:
        for seed in range(num_seeds):
            _, info = env.reset(seed=seed)
            count = int(info["object_count"])
            counts[count] += 1
            frame = env.render()
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"{env_name}.render() returned {type(frame).__name__}")
            frames.append(_annotate(frame, env_name, seed, count))
            if (seed + 1) % 10 == 0 or seed + 1 == num_seeds:
                print(f"{env_name}: rendered {seed + 1}/{num_seeds}", flush=True)
    finally:
        env.close()

    gif_path = output_dir / f"{env_name}.gif"
    save_video(frames, gif_path, fps=fps)
    return {
        "gif": gif_path.name,
        "num_seeds": num_seeds,
        "sampled_counts": dict(sorted(counts.items())),
        "frame_shape": list(frames[0].shape),
        "size_bytes": gif_path.stat().st_size,
    }


def main() -> None:
    """Render all requested generalized 3D initial-state distributions."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generalized_3d_initial_states"),
    )
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--render-dpi", type=int, default=60)
    parser.add_argument("--configs", nargs="*", default=_DEFAULT_CONFIGS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        env_name: _render_config(
            env_name,
            args.output_dir,
            args.num_seeds,
            args.fps,
            args.render_dpi,
        )
        for env_name in args.configs
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
