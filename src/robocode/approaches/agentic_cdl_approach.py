"""An approach that uses a Claude agent to generate behavior-based solutions.

Instead of writing a monolithic policy, the agent is guided to decompose the
task into a fixed sequence of behaviors (CDL-style), each with an explicit
precondition, subgoal, and a feedforward policy body.  The agent must:

1.  Reason about the high-level behavior decomposition first.
2.  Implement and test each behavior in isolation before composing them.
3.  Chain the behaviours into a final ``GeneratedApproach``.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TypeVar

from gymnasium.spaces import Space
from omegaconf import DictConfig
from relational_structs.spaces import ObjectCentricStateSpace

from robocode import prompts
from robocode.approaches.base_approach import BaseApproach
from robocode.primitive_descriptions import format_primitives_description
from robocode.primitives import blackbox_primitive_manifest
from robocode.utils.apptainer_sandbox import ApptainerSandboxConfig
from robocode.utils.backends import create_backend
from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    DockerSandboxConfig,
)
from robocode.utils.env_server import (
    ENV_CLIENT_SRC,
    env_server_running,
    serialize_space,
    write_env_spaces,
)
from robocode.utils.episode import load_generated_approach
from robocode.utils.rate_limit import run_with_rate_limit_retry
from robocode.utils.sandbox import SandboxConfig
from robocode.utils.sandbox_types import resolve_container_backend

logger = logging.getLogger(__name__)

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class AgenticCDLApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that uses a Claude agent to write behavior-decomposed code."""

    def __init__(
        self,
        action_space: Space[_ActType],
        observation_space: Space[_ObsType],
        seed: int,
        primitives: dict[str, Callable[..., Any]],
        backend: DictConfig,  # Hydra backend config (backend name, model, etc.)
        env_description_path: str | None = None,
        max_budget_usd: float = 5.0,
        max_turns: int = 0,
        output_dir: str = ".",
        load_dir: str | None = None,
        use_docker: bool = False,
        container_backend: str | None = None,
        geometry_prompt: bool = True,
        mcp_tools: tuple[str, ...] = (),
        max_output_tokens: int = 16384,
        autocompact_pct: int = 80,
        env_name: str | None = None,
        blackbox: bool = False,
        env_cfg: str | None = None,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_space,
            observation_space,
            seed,
            primitives,
            env_description_path,
            **kwargs,
        )
        self._backend_cfg = backend
        self._backend = create_backend(backend)
        self._model = backend["model"]
        self._max_budget_usd = max_budget_usd
        self._max_turns = max_turns
        self._output_dir = Path(output_dir)
        self._load_dir = Path(load_dir) if load_dir is not None else None
        self._container_backend = resolve_container_backend(
            container_backend, use_docker
        )
        self._geometry_prompt = geometry_prompt
        self._mcp_tools = mcp_tools
        self._max_output_tokens = max_output_tokens
        self._autocompact_pct = autocompact_pct
        self._env_name = env_name
        self._blackbox = blackbox
        self._env_cfg = env_cfg
        self._max_steps = max_steps
        if blackbox:
            if env_cfg is None:
                raise ValueError("blackbox mode requires env_cfg")
            # Fail fast on spaces the blackbox protocol cannot serialize.
            serialize_space(action_space)
            serialize_space(observation_space)
            if self._container_backend == "local":
                logger.warning(
                    "blackbox with the local backend is best-effort only: "
                    "the OS sandbox cannot prevent reading env source from "
                    "the host filesystem"
                )
        # A variable-count env hands the policy an ObjectCentricState; the prompt then
        # describes reading typed objects instead of indexing a flat vector.
        self._object_centric = isinstance(observation_space, ObjectCentricStateSpace)
        self._generated: Any = None
        self.total_cost_usd: float | None = None

    def train(self) -> None:  # noqa: C901 — mirrors AgenticApproach.train
        """Generate the behavior-based approach via a sandboxed Claude agent."""
        if self._load_dir is not None:
            approach_file = self._load_dir / "sandbox" / "approach.py"
            if not approach_file.exists():
                raise FileNotFoundError(f"No approach file at {approach_file}")
            self._load_generated(approach_file)
            return

        sandbox_dir = self._output_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Seed the sandbox with the Behavior base class so the agent can
        # inherit from it.
        behavior_src = (
            Path(__file__).resolve().parent.parent / "primitives" / "behavior.py"
        )
        init_files: dict[str, Path] = {"behavior.py": behavior_src}
        if self._blackbox:
            init_files["env_client.py"] = ENV_CLIENT_SRC

        # If env-specific helper files exist, copy them into the sandbox so
        # the agent starts with correct obs/act helpers instead of writing
        # them from scratch. Skipped in blackbox mode: these helpers spell out
        # the observation layout (feature names and indices), which is exactly
        # what the agent must discover empirically when the source is withheld.
        has_initial_helpers = False
        if self._env_name is not None and not self._blackbox:
            helpers_dir = (
                Path(__file__).resolve().parent.parent / "primitives" / self._env_name
            )
            for helper_name in ("obs_helpers.py", "act_helpers.py"):
                helper_path = helpers_dir / helper_name
                if helper_path.exists():
                    init_files[helper_name] = helper_path
                    has_initial_helpers = True

        # Build primitives description (shared with AgenticApproach; the blackbox flag
        # appends per-primitive black-box notes, e.g. how to feed the CRV planners the
        # state -- devectorized for a vector env, passed directly for object-centric).
        primitives_desc = format_primitives_description(
            list(self._primitives),
            blackbox=self._blackbox,
            object_centric=self._object_centric,
        )

        primitives_desc += prompts.build_mcp_tool_lines(
            mcp_tools=self._mcp_tools,
            backend_name=self._backend_cfg["backend"],
            blackbox=self._blackbox,
            object_centric=self._object_centric,
        )

        python_exe = (
            DOCKER_PYTHON if self._container_backend != "local" else sys.executable
        )
        interface_spec = prompts.build_interface_spec(
            class_interface=prompts.CDL_CLASS_INTERFACE,
            run_commands=prompts.CDL_RUN_COMMANDS,
            python_executable=python_exe,
            primitives_description=primitives_desc,
            blackbox=self._blackbox,
            object_centric=self._object_centric,
        )

        env_description: str | None = None
        if self._env_description_path is not None:
            env_description = Path(self._env_description_path).read_text(
                encoding="utf-8"
            )
        prompt = prompts.build_cdl_prompt(
            blackbox=self._blackbox,
            interface_spec=interface_spec,
            geometry=self._geometry_prompt,
            env_description=env_description,
            has_initial_helpers=has_initial_helpers,
            object_centric=self._object_centric,
        )

        system_prompt = prompts.build_system_prompt(
            intro=(prompts.CDL_INTRO_BLACKBOX if self._blackbox else prompts.CDL_INTRO),
            blackbox=self._blackbox,
            backend_name=self._backend_cfg["backend"],
            mcp_tools=self._mcp_tools,
            object_centric=self._object_centric,
        )

        docker_config: DockerSandboxConfig | None = None
        apptainer_config: ApptainerSandboxConfig | None = None
        config: SandboxConfig | None = None
        if self._container_backend == "docker":
            docker_config = DockerSandboxConfig(
                sandbox_dir=sandbox_dir,
                init_files=init_files,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=system_prompt,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
                max_turns=self._max_turns,
                primitive_names=tuple(self._primitives),
                mcp_tools=self._mcp_tools,
                max_output_tokens=self._max_output_tokens,
                autocompact_pct=self._autocompact_pct,
                blackbox=self._blackbox,
            )
            sandbox_logger = logging.getLogger("robocode.utils.docker_sandbox")
        elif self._container_backend == "apptainer":
            apptainer_config = ApptainerSandboxConfig(
                sandbox_dir=sandbox_dir,
                init_files=init_files,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=system_prompt,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
                max_turns=self._max_turns,
                primitive_names=tuple(self._primitives),
                mcp_tools=self._mcp_tools,
                max_output_tokens=self._max_output_tokens,
                autocompact_pct=self._autocompact_pct,
                blackbox=self._blackbox,
            )
            sandbox_logger = logging.getLogger("robocode.utils.apptainer_sandbox")
        else:
            config = SandboxConfig(
                sandbox_dir=sandbox_dir,
                init_files=init_files,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=system_prompt,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
                max_turns=self._max_turns,
                mcp_tools=self._mcp_tools,
                max_output_tokens=self._max_output_tokens,
                autocompact_pct=self._autocompact_pct,
                blackbox=self._blackbox,
            )
            sandbox_logger = logging.getLogger("robocode.utils.sandbox")

        log_path = sandbox_dir / "agent_log.txt"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        sandbox_logger.addHandler(file_handler)
        try:
            with ExitStack() as stack:
                if self._blackbox:
                    assert self._env_cfg is not None  # validated in __init__
                    port, token = stack.enter_context(
                        env_server_running(self._env_cfg, sandbox_dir)
                    )
                    write_env_spaces(
                        sandbox_dir,
                        container_backend=self._container_backend,
                        port=port,
                        token=token,
                        observation_space=self._state_space,
                        action_space=self._action_space,
                        max_steps=self._max_steps,
                        primitives_manifest=blackbox_primitive_manifest(
                            list(self._primitives)
                        ),
                    )
                result = run_with_rate_limit_retry(
                    docker_config,
                    config,
                    backend=self._backend,
                    apptainer_config=apptainer_config,
                )
        finally:
            sandbox_logger.removeHandler(file_handler)
            file_handler.close()

        self.total_cost_usd = result.total_cost_usd

        if result.success and result.output_file is not None:
            self._load_generated(result.output_file)
        else:
            logger.warning("Agent failed to generate approach: %s", result.error)

    def _load_generated(self, path: Path) -> None:
        """Load a GeneratedApproach class from the given file."""
        self._generated = load_generated_approach(
            path, self._action_space, self._state_space, self._primitives
        )

    def reset(self, state: _ObsType, info: dict[str, Any]) -> None:
        """Start a new episode."""
        super().reset(state, info)
        if self._generated is not None:
            self._generated.reset(state, info)

    def update(
        self,
        state: _ObsType,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Record the reward and next state following an action."""
        super().update(state, reward, done, info)
        if self._generated is not None and hasattr(self._generated, "update"):
            self._generated.update(state, reward, done, info)

    def _get_action(self) -> _ActType:
        if self._generated is not None:
            try:
                return self._generated.get_action(self._last_state)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Generated approach failed, using random")
        return self._action_space.sample()
