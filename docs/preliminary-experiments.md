# Preliminary experiments and alternative configurations

These paths remain in the repository for future experiments and development. **They are not used in the current paper**, except where this guide explicitly links to a paper method. This is an inventory of implemented/configured paths, not a claim that every provider/model or optional dependency has been rerun recently. Installation and the main paper protocol are in the [README](../README.md).

## Non-strict black box

Early experiments used **non-strict (legacy) black box**: environment source was withheld, but the agent could still use installed robotics/geometry libraries and environment helpers. The paper uses **strict black box** to study programs built from simulator interaction without that additional scaffolding.

| | Non-strict black box (preliminary) | Strict black box (paper) |
| --- | --- | --- |
| Access flags | `approach.blackbox=true approach.blackbox_strict=false` | `approach.blackbox=true approach.blackbox_strict=true` |
| Environment source | Withheld | Withheld |
| Program dependencies | Installed libraries remain available | Python standard library, NumPy, SciPy |
| Environment interface | `reset`/`step`, rendering, and helpers such as state access and observation conversion; configured primitives can also be exposed | `reset`/`step` and rendering; no state snapshots, state setting, or helper calls |
| Injected primitives | Selected by `primitive_level` | `primitive_level=none` |
| Container image | Regular image (`bash docker/build.sh`) | Strict image (`bash docker/build_strict_blackbox.sh`) |

Setting `primitive_level=none` in non-strict mode only removes injected primitives; it does not remove installed dependencies or restrict the environment interface to the strict surface. To reproduce the paper's black-box condition, use all three strict settings above. See [black-box access and protocol](blackbox.md) for implementation details.

## Alternative agent backends and models

Use `approach/backend=<preset>` with an agentic approach. The paper presets are `claude_opus5`, `codex_gpt56sol`, and `codex_gpt6`; all other checked-in presets are listed below. A configured model ID is not a guarantee of continuing provider availability.

| Preset | Backend | Configured model |
| --- | --- | --- |
| [claude_haiku45](../experiments/conf/approach/backend/claude_haiku45.yaml) | claude | `claude-haiku-4-5-20251001` |
| [claude_ollama_qwen](../experiments/conf/approach/backend/claude_ollama_qwen.yaml) | claude | `qwen3.5` |
| [claude_opus48](../experiments/conf/approach/backend/claude_opus48.yaml) | claude | `claude-opus-4-8` |
| [claude_sonnet46](../experiments/conf/approach/backend/claude_sonnet46.yaml) | claude | `claude-sonnet-4-6` |
| [claude_sonnet5](../experiments/conf/approach/backend/claude_sonnet5.yaml) | claude | `claude-sonnet-5` |
| [opencode_gpt4omini](../experiments/conf/approach/backend/opencode_gpt4omini.yaml) | opencode | `openai/gpt-4o-mini` |
| [opencode_gpt54](../experiments/conf/approach/backend/opencode_gpt54.yaml) | opencode | `openai/gpt-5.4` |
| [opencode_gpt5nano](../experiments/conf/approach/backend/opencode_gpt5nano.yaml) | opencode | `openai/gpt-5-nano` |
| [opencode_qwen](../experiments/conf/approach/backend/opencode_qwen.yaml) | opencode | `ollama/qwen3:0.6b` |

### OpenCode

[OpenCode](https://opencode.ai/docs/) is retained as a multi-provider agent backend. It is installed inside both images; for host use, `npm install -g opencode-ai` installs the CLI. Keep it on `PATH` (or set `ROBOCODE_OPENCODE_CMD`). Authenticate with the provider's supported API-key environment variable or OpenCode's auth store. Docker forwards supported provider credentials and mounts the OpenCode auth store. The isolated Apptainer backend rejects OpenCode.

For a non-paper Docker run, reuse the README's experiment command and replace its backend with `approach/backend=opencode_gpt54` (configured with `variant: low`), `opencode_gpt4omini`, or `opencode_gpt5nano`. Keep the desired access flags explicit. Provider/network support must match the sandbox's configured endpoints; changing a model string alone does not add support to a network broker.

### Ollama and vLLM

[Ollama](https://ollama.com/) and [vLLM](https://docs.vllm.ai/) are optional model servers, not paper prerequisites. Install them separately from the experiment virtualenv. `claude_ollama_qwen` expects `qwen3.5` at `http://localhost:11434`; `opencode_qwen` names `ollama/qwen3:0.6b`. Pull the matching Ollama model, for example `ollama pull qwen3.5`.

The local-model hooks are implemented in the backend code, but a host loopback server must be reachable from the chosen container transport. The current isolated Apptainer transport rejects custom model endpoints. Do not assume a local-server preset is interchangeable with a paper backend under strict networking. The retained low-level `local` sandbox allows host reads, and the experiment runner rejects `container_backend=local` for generated-code methods to protect the evaluation seed.

## Alternative completion providers

`llm_genplan` and `best_of_k` select `approach/completion=<preset>`, separate from the agentic backend group. The paper uses `cli_opus5`. Other checked-in choices are:

| Preset | Provider | Configured model |
| --- | --- | --- |
| [anthropic_opus](../experiments/conf/approach/completion/anthropic_opus.yaml) | anthropic | `claude-opus-4-8` |
| [anthropic_sonnet](../experiments/conf/approach/completion/anthropic_sonnet.yaml) | anthropic | `claude-sonnet-4-6` |
| [cli_claude](../experiments/conf/approach/completion/cli_claude.yaml) | cli | `sonnet` |
| [cli_opus48](../experiments/conf/approach/completion/cli_opus48.yaml) | cli | `claude-opus-4-8` |
| [cli_sonnet46](../experiments/conf/approach/completion/cli_sonnet46.yaml) | cli | `claude-sonnet-4-6` |
| [cli_sonnet5](../experiments/conf/approach/completion/cli_sonnet5.yaml) | cli | `claude-sonnet-5` |
| [ollama_qwen](../experiments/conf/approach/completion/ollama_qwen.yaml) | openai_compatible | `qwen3.6` |
| [vllm](../experiments/conf/approach/completion/vllm.yaml) | openai_compatible | `Qwen/Qwen3.6-35B-A3B` |

The `cli_claude` preset uses the moving `sonnet` alias; the model-specific CLI presets name full IDs. `anthropic_opus` is Opus **4.8**, not the paper's Opus 5, and the `anthropic_*` presets need `ANTHROPIC_API_KEY`. Their dollar accounting uses configured token-price estimates. The Ollama completion endpoint is `http://localhost:11434/v1`; vLLM's is `http://localhost:8000/v1`. Local completion providers do not report dollar cost, so retain a finite `max_debug_attempts` or `max_generation_steps` instead of relying only on a dollar limit. The source-access Docker image is used for completion-based generation; Apptainer transport is unsupported.

## Preliminary approaches and primitive variants

| Choice | Retained behavior and scope |
| --- | --- |
| `approach=agentic_cdl` | Behavior decomposition with preconditions; the constructor currently emits a deprecation warning. Code and prompts are retained. It supports source access and legacy black box, not `blackbox_strict`. |
| `approach=agentic_per_instance` | A new synthesis problem for each episode, sharing a global budget; optionally cap each episode with `approach.max_budget_per_instance_usd`. It receives the episode seed. This differs from one frozen generalized policy. |
| `approach=best_of_k` | Independent candidate generation without feedback; selects the best validation score. Defaults to five candidates and $20, no chain of thought. It is not the paper's One-shot. |
| `approach=random` | Random-action smoke baseline; no model credentials or synthesis container required. |
| `approach=oracle` | Handwritten solvability/reference policies for the choices in [ORACLE_TARGETS](../src/robocode/approaches/oracle_approach.py). These are separate from the paper planner baselines. |
| `primitive_level=low_level` | Injects `check_action_collision` and `BiRRT`; outside the paper's no-primitives condition. |
| `primitive_level=bilevel` | Injects benchmark planning models; requires the `bilevel` extra and a supported environment. |
| `approach.blackbox=true approach.blackbox_strict=false` | [Non-strict black box](#non-strict-black-box); withholds environment source but retains libraries and helpers. |
| `approach.geometry_prompt=true` | Optional geometry prompt for compatible agentic approaches. |
| `approach.modular_code_prompt=true`, `approach.token_budget_prompt=true` | Optional agentic/per-instance prompt variants; leave false for the main commands. |

Primitive definitions are in [primitive_level](../experiments/conf/primitive_level); helper code and handwritten policies remain under [primitives](../src/robocode/primitives) and [oracles](../src/robocode/oracles). Strict black box requires `primitive_level=none`. The `bilevel_models` primitive is not compatible with black-box access.

## Maze and non-paper environment configs

All 28 paper configs are mapped in the [README](../README.md#environment-map). The table below accounts for **every other checked-in environment config**, including fixed-difficulty variants of paper families. These are useful smoke tests or earlier experiment conditions, not additional paper environments.

| Config | Scope |
| --- | --- |
| [clutteredretrieval2d_easy](../experiments/conf/environment/clutteredretrieval2d_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [clutteredretrieval2d_hard](../experiments/conf/environment/clutteredretrieval2d_hard.yaml) | Fixed-difficulty/count variant of a paper family |
| [clutteredretrieval2d_medium](../experiments/conf/environment/clutteredretrieval2d_medium.yaml) | Fixed-difficulty/count variant of a paper family |
| [clutteredstorage2d_easy](../experiments/conf/environment/clutteredstorage2d_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [clutteredstorage2d_hard](../experiments/conf/environment/clutteredstorage2d_hard.yaml) | Fixed-difficulty/count variant of a paper family |
| [clutteredstorage2d_medium](../experiments/conf/environment/clutteredstorage2d_medium.yaml) | Fixed-difficulty/count variant of a paper family |
| [constrainedcupboard3d_easy](../experiments/conf/environment/constrainedcupboard3d_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [large_maze](../experiments/conf/environment/large_maze.yaml) | Discrete maze demo |
| [motion2d_easy](../experiments/conf/environment/motion2d_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [motion2d_hard](../experiments/conf/environment/motion2d_hard.yaml) | Fixed-difficulty/count variant of a paper family |
| [motion2d_medium](../experiments/conf/environment/motion2d_medium.yaml) | Fixed-difficulty/count variant of a paper family |
| [obstruction2d_easy](../experiments/conf/environment/obstruction2d_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [obstruction2d_hard](../experiments/conf/environment/obstruction2d_hard.yaml) | Fixed-difficulty/count variant of a paper family |
| [obstruction2d_medium](../experiments/conf/environment/obstruction2d_medium.yaml) | Fixed-difficulty/count variant of a paper family |
| [obstruction3d_easy](../experiments/conf/environment/obstruction3d_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [obstruction3d_hard](../experiments/conf/environment/obstruction3d_hard.yaml) | Fixed-difficulty/count variant of a paper family |
| [obstruction3d_medium](../experiments/conf/environment/obstruction3d_medium.yaml) | Fixed-difficulty/count variant of a paper family |
| [packing3d_easy](../experiments/conf/environment/packing3d_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [packing3d_hard](../experiments/conf/environment/packing3d_hard.yaml) | Fixed-difficulty/count variant of a paper family |
| [packing3d_medium](../experiments/conf/environment/packing3d_medium.yaml) | Fixed-difficulty/count variant of a paper family |
| [pr2blocked_easy](../experiments/conf/environment/pr2blocked_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [pr2blocked_hard](../experiments/conf/environment/pr2blocked_hard.yaml) | Fixed-difficulty/count variant of a paper family |
| [pr2blocked_medium](../experiments/conf/environment/pr2blocked_medium.yaml) | Fixed-difficulty/count variant of a paper family |
| [pr2packed_easy](../experiments/conf/environment/pr2packed_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [pr2packed_hard](../experiments/conf/environment/pr2packed_hard.yaml) | Fixed-difficulty/count variant of a paper family |
| [pr2packed_medium](../experiments/conf/environment/pr2packed_medium.yaml) | Fixed-difficulty/count variant of a paper family |
| [rovers_easy](../experiments/conf/environment/rovers_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [rovers_hard](../experiments/conf/environment/rovers_hard.yaml) | Fixed-difficulty/count variant of a paper family |
| [rovers_medium](../experiments/conf/environment/rovers_medium.yaml) | Fixed-difficulty/count variant of a paper family |
| [shelf3d_easy](../experiments/conf/environment/shelf3d_easy.yaml) | Kinematic Shelf (separate from paper Dynamic3D Shelf) |
| [shelf3d_generalized](../experiments/conf/environment/shelf3d_generalized.yaml) | Kinematic Shelf (separate from paper Dynamic3D Shelf) |
| [shelf3d_hard](../experiments/conf/environment/shelf3d_hard.yaml) | Kinematic Shelf (separate from paper Dynamic3D Shelf) |
| [shelf3d_medium](../experiments/conf/environment/shelf3d_medium.yaml) | Kinematic Shelf (separate from paper Dynamic3D Shelf) |
| [small_maze](../experiments/conf/environment/small_maze.yaml) | Discrete maze demo |
| [stickbutton2d_easy](../experiments/conf/environment/stickbutton2d_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [stickbutton2d_hard](../experiments/conf/environment/stickbutton2d_hard.yaml) | Fixed-difficulty/count variant of a paper family |
| [stickbutton2d_medium](../experiments/conf/environment/stickbutton2d_medium.yaml) | Fixed-difficulty/count variant of a paper family |
| [transport3d_easy](../experiments/conf/environment/transport3d_easy.yaml) | Fixed-difficulty/count variant of a paper family |
| [transport3d_hard](../experiments/conf/environment/transport3d_hard.yaml) | Fixed-difficulty/count variant of a paper family |

For a no-model smoke run after setting `EVAL_SEED`:

```bash
python experiments/run_experiment.py \
  approach=random environment=small_maze primitive_level=none \
  replicate_seed=0 eval_seed="$EVAL_SEED" num_eval_tasks=10 eval_timeout=60
```

Maze agentic runs use the same runner with `approach=agentic`; choose a container, authenticate, and build its image first. The old maze policy listing and embedded agent transcript have been removed from the main README; the environment implementation/configs and tests remain.

The fixed-count PR2 environments use flat observations; their generalized variants use object-centric states. Packing places all blocks on the plate; Blocked retrieves any green block, so more spare green blocks can make that task easier.

## LIBERO-PRO

[LIBERO-PRO](https://github.com/uynitsuj/LIBERO-PRO) is a Franka tabletop manipulation benchmark (goal / spatial / object / 10-task mixes plus OOD and perturbation variants) built on MuJoCo via robosuite. It is vendored as a submodule under `third-party/LIBERO-PRO/` and gated behind the optional `libero` extra — it is **not** installed by default because it pins old upstreams (`robosuite==1.4.0`, `gym==0.25.2`, `robomimic==0.2.0`, `bddl==1.0.1`) and drags in a CUDA-enabled torch.

`install.sh` passes `--no-extra libero`, so the default install skips it entirely.

**Linux only.**
The extra cannot be installed on macOS: `robomimic 0.2.0` depends on `egl-probe`, which compiles an EGL loader, and EGL has no macOS implementation.
Use a Linux host for this optional setup; the standard sandbox image does not install the `libero` extra.

Install (into the same venv as the rest of robocode):

```bash
sudo apt-get install -y libegl1 libgl1 cmake  # EGL/GL runtime for MuJoCo; cmake builds egl-probe
uv sync --all-extras --dev                    # ~60 extra Python packages, several GB
```

First use of the `libero` package runs an interactive `input()` prompt asking where to store datasets; the test harness writes `~/.libero/config.yaml` automatically. If you hit the prompt manually, answer `N` — the default paths are fine for env rollouts (pre-recorded demos are not required).

List available benchmark suites:

```python
from libero import benchmark
print(list(benchmark.get_benchmark_dict().keys()))  # available suites in the pinned version
```

Minimal rollout on `libero_goal` task 0:

```python
from libero import benchmark
from libero.envs import OffScreenRenderEnv

task_suite = benchmark.get_benchmark_dict()["libero_goal"]()
bddl = task_suite.get_task_bddl_file_path(0)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(0)
obs = env.reset()   # dict with agentview_image, robot state, per-object poses, ...
obs, reward, done, info = env.step([0.0] * 7)
env.close()
```

Smoke tests live at `tests/environments/test_libero.py` (benchmark dict + rollout); they skip cleanly if the extra isn't installed.

Note on OpenGL: LIBERO's MuJoCo needs to coexist in-process with kinder's pybullet. `src/robocode/environments/kinder_geom2d_env.py` and `kinder_geom3d_env.py` pin `MUJOCO_GL=egl` / `PYOPENGL_PLATFORM=egl` before kinder loads so PyOpenGL latches to the EGL platform — without this, later robosuite imports in the same process fail with `'NoneType' object has no attribute 'glGetError'`. If you see that error, confirm `libegl1` is installed.


This is an optional upstream rollout/test path; there is no first-party LIBERO Hydra environment config in the current experiment suite.

## CaP-X

The [CaP-X submodule](../third-party/cap-x) is retained for exploratory work and has its own setup, environment configs, and web UI. Follow its upstream README after initializing submodules. It is not wired into a first-party `environment=...` or `approach=...` choice here, and is not part of the current paper. Its third-party documentation and dependencies are left intact.
