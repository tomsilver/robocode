"""Count-parameterized access to the registered ConstrainedCupboard3D tasks."""

from pathlib import Path
from typing import Any

from kinder.envs.dynamic3d import envs as dynamic3d_envs
from kinder.envs.dynamic3d.envs import TidyBot3DEnv


class ConstrainedCupboard3DEnv(TidyBot3DEnv):
    """Select a ConstrainedCupboard3D task by its number of rods."""

    supported_counts = frozenset({1, 2, 6})

    def __init__(self, num_objects: int = 1, **kwargs: Any) -> None:
        if num_objects not in self.supported_counts:
            raise ValueError(
                f"ConstrainedCupboard3D supports counts "
                f"{sorted(self.supported_counts)}, got {num_objects}"
            )
        tasks_dir = Path(dynamic3d_envs.__file__).parent / "tasks"
        task_path = (
            tasks_dir
            / "ConstrainedCupboard3D"
            / f"ConstrainedCupboard3D-o{num_objects}.json"
        )
        super().__init__(
            num_objects=num_objects,
            task_config_path=str(task_path),
            scene_render_camera="task_view",
            **kwargs,
        )
