"""Shared base for approaches that run a sandboxed coding agent.

``GeneratedProgramApproach`` holds the machinery common to the generalized
``AgenticApproach`` (one ``train()`` run produces a policy evaluated over every
seed) and the per-instance ``AgenticPerInstanceApproach`` (a fresh sandbox per
eval seed): the constructor config, the sandbox-running core (``_run_sandbox``),
loading the produced ``GeneratedApproach``, and delegating the episode lifecycle
to that loaded program.

Subclasses choose the *lifecycle*, not the implementation: ``AgenticApproach``
implements ``train()``; ``AgenticPerInstanceApproach`` implements
``solve_instance()``. Keeping the lifecycles in separate classes (rather than one
class that does both) is deliberate, so the generalized baseline cannot be broken
by changes to the per-instance path.
"""

import logging
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TypeVar

from gymnasium.spaces import Space
from omegaconf import DictConfig

from robocode import prompts
from robocode.approaches.base_approach import BaseApproach
from robocode.primitives import (
    blackbox_primitive_manifest,
    format_primitives_description,
)
from robocode.utils.apptainer_sandbox import ApptainerSandboxConfig
from robocode.utils.backends import create_backend
from robocode.utils.docker_sandbox import DOCKER_PYTHON, DockerSandboxConfig
from robocode.utils.env_server import (
    ENV_CLIENT_SRC,
    env_server_running,
    serialize_space,
    write_env_spaces,
)
from robocode.utils.episode import load_generated_approach
from robocode.utils.rate_limit import run_with_rate_limit_retry
from robocode.utils.sandbox import SandboxConfig
from robocode.utils.sandbox_types import SandboxResult, resolve_container_backend

logger = logging.getLogger(__name__)

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class GeneratedProgramApproach(BaseApproach[_ObsType, _ActType]):
    """Base for approaches whose policy is a sandbox-agent-written program."""

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
        modular_code_prompt: bool = False,
        mcp_tools: tuple[str, ...] = (),
        max_output_tokens: int = 16384,
        autocompact_pct: int = 80,
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
        self._modular_code_prompt = modular_code_prompt
        self._mcp_tools = mcp_tools
        self._max_output_tokens = max_output_tokens
        self._autocompact_pct = autocompact_pct
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
        self._generated: Any = None
        self.total_cost_usd: float | None = None

    def _build_agentic_prompts(
        self, *, per_instance_seed: int | None = None
    ) -> tuple[str, str, dict[str, Path]]:
        """Assemble the (task prompt, system prompt, init files) for one run.

        Shared by the generalized ``train()`` (``per_instance_seed=None``) and
        per-instance ``solve_instance()`` (a concrete seed); the only difference
        is the seed threaded into the task prompt. If we have an env description
        it is inlined; otherwise the agent is asked to read source files.
        """
        primitives_desc = format_primitives_description(
            list(self._primitives), blackbox=self._blackbox
        )
        primitives_desc += prompts.build_mcp_tool_lines(
            mcp_tools=self._mcp_tools,
            backend_name=self._backend_cfg["backend"],
            blackbox=self._blackbox,
        )

        python_exe = (
            DOCKER_PYTHON if self._container_backend != "local" else sys.executable
        )
        interface_spec = prompts.build_interface_spec(
            class_interface=prompts.AGENTIC_CLASS_INTERFACE,
            run_commands=prompts.AGENTIC_RUN_COMMANDS,
            python_executable=python_exe,
            primitives_description=primitives_desc,
            blackbox=self._blackbox,
        )

        env_description: str | None = None
        if self._env_description_path is not None:
            env_description = Path(self._env_description_path).read_text(
                encoding="utf-8"
            )
        prompt = prompts.build_agentic_prompt(
            blackbox=self._blackbox,
            interface_spec=interface_spec,
            geometry=self._geometry_prompt,
            modular_code=self._modular_code_prompt,
            env_description=env_description,
            per_instance_seed=per_instance_seed,
        )

        system_prompt = prompts.build_system_prompt(
            intro=(
                prompts.AGENTIC_INTRO_BLACKBOX
                if self._blackbox
                else prompts.AGENTIC_INTRO
            ),
            blackbox=self._blackbox,
            backend_name=self._backend_cfg["backend"],
            mcp_tools=self._mcp_tools,
        )
        init_files: dict[str, Path] = {}
        if self._blackbox:
            init_files["env_client.py"] = ENV_CLIENT_SRC
        return prompt, system_prompt, init_files

    def _run_sandbox(
        self,
        *,
        sandbox_dir: Path,
        prompt: str,
        system_prompt: str,
        max_budget_usd: float,
        init_files: dict[str, Path],
    ) -> SandboxResult:
        """Build the sandbox config, run the agent, and return its result.

        Shared by the generalized ``train()`` and the per-instance
        ``solve_instance()``: only the prompts, sandbox dir, and budget differ
        between callers. When ``self._blackbox`` is set, the live env server runs
        for the duration of the agent session.
        """
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
                max_budget_usd=max_budget_usd,
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
                max_budget_usd=max_budget_usd,
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
                max_budget_usd=max_budget_usd,
                max_turns=self._max_turns,
                mcp_tools=self._mcp_tools,
                max_output_tokens=self._max_output_tokens,
                autocompact_pct=self._autocompact_pct,
                blackbox=self._blackbox,
            )
            sandbox_logger = logging.getLogger("robocode.utils.sandbox")

        # Write agent logs to a file in the sandbox directory.
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
        return result

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
        # Never silently fall back to random: a random eval would propagate a
        # misleading 0 through the results as if the generated approach had been
        # measured. Fail loudly instead. Use approach=random for a random
        # baseline. A runtime error in the generated approach also propagates.
        if self._generated is None:
            raise RuntimeError(
                "No generated approach is loaded (the agent did not produce a "
                "usable approach.py). Refusing to evaluate a silent random "
                "policy; use approach=random for a random baseline."
            )
        return self._generated.get_action(self._last_state)
