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
    frame: NDArray[np.uint8],
    env_name: str,
    phase: str,
    index: int,
    total: int,
    seed: int,
    count: int,
) -> NDArray[np.uint8]:
    rgb = frame[..., :3]
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    label = (
        f"{env_name} | {phase} {index:03d}/{total:03d} | "
        f"seed={seed} | object_count={count}"
    )
    _, _, _, text_height = draw.textbbox((0, 0), label)
    draw.rectangle((0, 0, image.width, text_height + 8), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return np.asarray(image)


def _render_config(
    env_name: str,
    output_dir: Path,
    num_train_seeds: int,
    num_eval_seeds: int,
    eval_seed: int,
    fps: int,
    render_dpi: int,
) -> dict[str, Any]:
    config = OmegaConf.load(_CONFIG_DIR / f"{env_name}.yaml")
    config.render_dpi = render_dpi
    env = instantiate(config)
    frames: list[NDArray[np.uint8]] = []
    phase_counts: dict[str, Counter[int]] = {
        "train": Counter(),
        "eval": Counter(),
    }
    try:
        train_seeds = list(range(num_train_seeds))
        eval_rng = np.random.default_rng(eval_seed)
        eval_seeds = [int(eval_rng.integers(0, 2**63)) for _ in range(num_eval_seeds)]
        eval_counts = [
            env.eval_counts[i % len(env.eval_counts)] for i in range(num_eval_seeds)
        ]
        phases = (
            ("train", train_seeds, [None] * num_train_seeds),
            ("eval", eval_seeds, eval_counts),
        )
        for phase, seeds, pinned_counts in phases:
            total = len(seeds)
            for index, (seed, pinned_count) in enumerate(
                zip(seeds, pinned_counts, strict=True), start=1
            ):
                options = (
                    None
                    if pinned_count is None
                    else {"object_count": int(pinned_count)}
                )
                _, info = env.reset(seed=seed, options=options)
                count = int(info["object_count"])
                phase_counts[phase][count] += 1
                frame = env.render()
                if not isinstance(frame, np.ndarray):
                    raise TypeError(
                        f"{env_name}.render() returned {type(frame).__name__}"
                    )
                frames.append(
                    _annotate(frame, env_name, phase, index, total, seed, count)
                )
                if index % 10 == 0 or index == total:
                    print(f"{env_name} {phase}: rendered {index}/{total}", flush=True)
    finally:
        env.close()

    gif_path = output_dir / f"{env_name}.gif"
    save_video(frames, gif_path, fps=fps)
    return {
        "gif": gif_path.name,
        "num_train_seeds": num_train_seeds,
        "num_eval_seeds": num_eval_seeds,
        "eval_seed": eval_seed,
        "train_counts": dict(sorted(phase_counts["train"].items())),
        "eval_counts": dict(sorted(phase_counts["eval"].items())),
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
    parser.add_argument("--num-train-seeds", type=int, default=100)
    parser.add_argument("--num-eval-seeds", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--render-dpi", type=int, default=60)
    parser.add_argument("--configs", nargs="*", default=_DEFAULT_CONFIGS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        env_name: _render_config(
            env_name,
            args.output_dir,
            args.num_train_seeds,
            args.num_eval_seeds,
            args.eval_seed,
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
