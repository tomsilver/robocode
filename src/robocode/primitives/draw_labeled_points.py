"""Draw labeled points onto an image."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")


def draw_labeled_points(
    image: NDArray[np.uint8],
    points: list[tuple[int, int, str]],
    marker_radius: int = 5,
    font_size: int = 12,
) -> NDArray[np.uint8]:
    """Return a copy of *image* with labeled markers drawn at each point.

    Parameters
    ----------
    image:
        H×W×3 RGB uint8 array.
    points:
        List of ``(pixel_x, pixel_y, label)`` tuples.
    marker_radius:
        Size of the circular marker (matplotlib marker size).
    font_size:
        Font size for the text labels.
    """
    if not points:
        return image.copy()

    h, w = image.shape[:2]
    dpi = 100.0
    fig, ax = plt.subplots(
        figsize=(w / dpi, h / dpi),
        dpi=dpi,
    )
    ax.imshow(image)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for px, py, label in points:
        ax.plot(
            px,
            py,
            "o",
            markersize=marker_radius,
            markeredgecolor="white",
            markerfacecolor="red",
        )
        ax.annotate(
            label,
            (px, py),
            textcoords="offset points",
            xytext=(marker_radius + 4, -(marker_radius + 4)),
            fontsize=font_size,
            color="white",
            bbox={"boxstyle": "round,pad=0.2", "fc": "black", "alpha": 0.6},
        )

    fig.canvas.draw()
    buf: NDArray[np.uint8] = np.asarray(
        fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
    )[..., :3].copy()
    plt.close(fig)

    # Resize to match original dimensions exactly (canvas rounding).
    if buf.shape[:2] != (h, w):
        from PIL import Image  # pylint: disable=import-outside-toplevel

        buf = np.asarray(
            Image.fromarray(buf).resize((w, h), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )

    return buf
