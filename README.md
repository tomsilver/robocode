# RoboCode

![workflow](https://github.com/tomsilver/robocode/actions/workflows/ci.yml/badge.svg)

Agents for robot physical reasoning.

Work in progress.

## Installation
```bash
git clone https://github.com/tomsilver/robocode.git
cd robocode
bash install.sh
```

This installs everything except the optional [LIBERO-PRO](#libero-pro-manipulation-benchmark-optional-extra) extra, which is opt-in.

### Python version

**Use Python 3.11.**
It is pinned in `.python-version`, so `uv` picks it up automatically and `install.sh` needs no flags.
If `uv` has no 3.11 on hand it downloads one.

3.11 is the only version actually exercised: the CI matrix is `["3.11"]`, and the Docker image installs `python3.11` and syncs against it.
3.12 resolves and would probably work, but nothing tests it, and a local/sandbox version split is a bad trade in a project where agent code runs in the container.
Moving to 3.12 means updating the CI matrix and the Dockerfile first.

3.13 and newer cannot work at all, because the `kindergarden` submodule declares `requires-python = ">=3.10,<3.13"`.
`robocode` mirrors that ceiling in its own `requires-python` so `uv` rejects a too-new interpreter up front instead of failing deep in a build.

### System prerequisites

The following tools are **not** installed by `install.sh` / `uv sync` and must be set up separately:

| Tool | Required for | Install |
|---|---|---|
| [Ollama](https://ollama.com/) | Local model serving (Claude + Ollama, OpenCode + Ollama) | `curl -fsSL https://ollama.com/install.sh \| sh` |
| [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) | `claude` backend (default) | `curl -fsSL https://claude.ai/install.sh \| bash` |
| [OpenCode CLI](https://opencode.ai) | `opencode` backend (multi-provider) | `curl -fsSL https://opencode.ai/install \| bash` |
| [vLLM](https://docs.vllm.ai/) | Serving models via OpenAI-compatible API | `pip install vllm` (in a separate env) |
| [Docker](https://www.docker.com/) | Docker sandbox (recommended for isolation) | See [Docker docs](https://docs.docker.com/get-docker/) |

For local model serving with Ollama, pull a model after installing:

```bash
ollama pull gemma4:31b
```

### Agent backend setup

The agentic approach supports two backends: **Claude Code CLI** (default) and **OpenCode** (for GPT, Gemini, open-source models via vLLM/Ollama, etc.).

#### Claude Code CLI (default)

The [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude`) is the default backend. Authenticate via one of:

- **Subscription (free usage):** `claude auth login`
- **API key:** set `ANTHROPIC_API_KEY` in your environment

Optionally set `ROBOCODE_CLAUDE_CMD` to point to a specific `claude` binary (defaults to `claude` on `PATH`).

The `model` parameter in `agentic.yaml` takes a full model ID. Override per-run with e.g. `approach/backend=claude_sonnet5`.

| Backend preset | Model ID |
|---|---|
| `claude_opus5` (default) | `claude-opus-5` |
| `claude_sonnet5` | `claude-sonnet-5` |
| `claude_opus48` | `claude-opus-4-8` |
| `claude_sonnet46` | `claude-sonnet-4-6` |
| `claude_haiku45` | `claude-haiku-4-5-20251001` |

See [Anthropic models overview](https://platform.claude.com/docs/en/about-claude/models/overview) for the full list.

#### OpenCode (multi-provider)

[OpenCode](https://opencode.ai) supports 75+ providers including OpenAI, Google, Anthropic, and local models served via Ollama or vLLM.

Install: `curl -fsSL https://opencode.ai/install | bash` (also pre-installed in the Docker image).

Authenticate with your provider:

```bash
# API key (set the appropriate env var for your provider)
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...

# Or use OpenCode's interactive auth
opencode providers login
```

Optionally set `ROBOCODE_OPENCODE_CMD` to point to a specific `opencode` binary.

Models use the `provider/model` format:

| Model | Provider |
|---|---|
| `openai/gpt-4o` | OpenAI |
| `google/gemini-2.5-pro` | Google |
| `anthropic/claude-sonnet-5` | Anthropic |
| `ollama/qwen3.5:latest` | Ollama (local) |

For local models (Ollama, vLLM), create an `opencode.json` config with your provider:

```json
{
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": { "qwen3.5:latest": { "name": "Qwen 3.5" } }
    }
  }
}
```

## Environments

All environments are available as Hydra configs via `environment=<config_name>`.

### Maze (discrete)

| Config | Description |
|---|---|
| `small_maze` | Small grid maze |
| `large_maze` | Large grid maze |

### 2D Kinematic (continuous, kinder geom2d)

| Config | Kinder ID | Difficulty |
|---|---|---|
| `motion2d_easy` | `kinder/Motion2D-p0-v0` | Easy (0 passages) |
| `motion2d_medium` | `kinder/Motion2D-p1-v0` | Medium (1 passage) |
| `motion2d_hard` | `kinder/Motion2D-p3-v0` | Hard (3 passages) |
| `obstruction2d_easy` | `kinder/Obstruction2D-o0-v0` | Easy (0 obstructions) |
| `obstruction2d_medium` | `kinder/Obstruction2D-o2-v0` | Medium (2 obstructions) |
| `obstruction2d_hard` | `kinder/Obstruction2D-o4-v0` | Hard (4 obstructions) |
| `clutteredretrieval2d_easy` | `kinder/ClutteredRetrieval2D-o1-v0` | Easy (1 obstruction) |
| `clutteredretrieval2d_medium` | `kinder/ClutteredRetrieval2D-o10-v0` | Medium (10 obstructions) |
| `clutteredretrieval2d_hard` | `kinder/ClutteredRetrieval2D-o25-v0` | Hard (25 obstructions) |
| `clutteredstorage2d_easy` | `kinder/ClutteredStorage2D-b1-v0` | Easy (1 block) |
| `clutteredstorage2d_medium` | `kinder/ClutteredStorage2D-b3-v0` | Medium (3 blocks) |
| `clutteredstorage2d_hard` | `kinder/ClutteredStorage2D-b7-v0` | Hard (7 blocks) |
| `stickbutton2d_easy` | `kinder/StickButton2D-b1-v0` | Easy (1 button) |
| `stickbutton2d_medium` | `kinder/StickButton2D-b3-v0` | Medium (3 buttons) |
| `stickbutton2d_hard` | `kinder/StickButton2D-b5-v0` | Hard (5 buttons) |
| `pushpullhook2d` | `kinder/PushPullHook2D-v0` | Single variant |

### 3D Kinematic (continuous, kinder geom3d)

| Config | Kinder ID | Difficulty |
|---|---|---|
| `obstruction3d_easy` | `kinder/Obstruction3D-o0-v0` | Easy (0 obstructions) |
| `obstruction3d_medium` | `kinder/Obstruction3D-o2-v0` | Medium (2 obstructions) |
| `obstruction3d_hard` | `kinder/Obstruction3D-o4-v0` | Hard (4 obstructions) |
| `shelf3d_easy` | `kinder/KinematicShelf3D-o1-v0` | Easy (1 cube) |
| `shelf3d_medium` | `kinder/KinematicShelf3D-o3-v0` | Medium (3 cubes) |
| `shelf3d_hard` | `kinder/KinematicShelf3D-o5-v0` | Hard (5 cubes) |
| `transport3d_easy` | `kinder/Transport3D-o1-v0` | Easy (1 cube) |
| `transport3d_hard` | `kinder/Transport3D-o2-v0` | Hard (2 cubes) |
| `packing3d_easy` | `kinder/Packing3D-p1-v0` | Easy (1 part) |
| `packing3d_medium` | `kinder/Packing3D-p2-v0` | Medium (2 parts) |
| `packing3d_hard` | `kinder/Packing3D-p3-v0` | Hard (3 parts) |

> **Realistic 3D backgrounds (optional):** dynamic3d / TidyBot envs (e.g. `ConstrainedCupboard3D`) render on a plain white background by default. For the realistic room scene (floor/wall textures), download the MimicLabs assets once (~1 GB, gitignored): `python third-party/kindergarden/scripts/download_mimiclabs_assets.py`, then set `scene_bg: mimiclabs-lab2` in the env config (already set for `constrainedcupboard3d_easy`).

### PR2 TAMP (continuous, ss-pybullet)

PDDLStream's [`packed`](https://github.com/caelan/pddlstream) benchmark, re-exposed as a
closed-loop gymnasium environment: a PR2 must pick every block off the table and place it
on a green plate. The scene and goal are carried over verbatim so instances line up with
the `packed -n` conditions in the [LLM-PDDLStream](https://github.com/jorge-a-mendez/llm-pddlstream)
paper, but the PDDL domain, the stream samplers, and the planner are deliberately left
behind: robocode's approaches synthesize their own task and motion planning.

| Config | Blocks | Observation |
|---|---|---|
| `pr2packed_easy` | 3 | `Box(72,)` |
| `pr2packed_medium` | 4 | `Box(83,)` |
| `pr2packed_hard` | 5 | `Box(94,)` |
| `pr2packed_generalized` | 1-5 (varies per reset) | `ObjectCentricState` |

`pr2packed_generalized` is the generalization axis: it keeps one backend per count and
returns object-centric observations, so a single frozen program spans every instance
size. It implements `VariableCountEnv` (`src/robocode/environments/variable_count.py`),
the shared contract the runner's count-sweep lifecycle keys off — `design_counts` are
what an approach is built against, `eval_counts` adds held-out larger instances, and
the results carry `by_count` and a design/held-out `count_regimes` split. The kinder
`VariableObjectCountEnv` implements the same contract.

The action space matches kinder's 3D mobile-manipulation envs exactly — `Box(11,)` of
delta base pose, delta arm joints, and gripper open/close — so approaches and prompts
written against `obstruction3d` and friends transfer without a new interface to learn.
Dynamics are kinematic: joints are set rather than servoed, and a motion that would put
the robot or a held block in collision is rejected. Closing the gripper attaches a block
rigidly to the tool frame, but only when the fingers are actually around it — hovering
above a block does not grasp it. Opening drops the held block straight down, and the drop
is refused if it would land overlapping something, leaving the block held. Reward is -1
per step and the episode terminates once every block rests on the plate **and no two
blocks overlap**: the plate is small enough to force a packing, so dropping them all at
one spot does not count.

**Verifying against stock PDDLStream.** `scripts/compare_pddlstream_rollout.py` rolls
episodes in our environment and replays every state visited into a scene built from
stock `examples.pybullet.tamp.problems.packed`, comparing the stock library's own
collision / placement / kinematics results against ours, and optionally rendering the
two side by side. The stock tree is pinned as a submodule but **not cloned by
default** — its nested submodule is a second copy of ss-pybullet at the commit we
already vendor (~590MB). Opt in with:

```bash
git submodule update --init --recursive third-party/pddlstream
python scripts/compare_pddlstream_rollout.py --policy oracle --render /tmp/cmp
```

Exit code 0 means every recorded value agreed to 1e-9; 1 prints a per-value diff.

**Oracle.** Run it with `approach=oracle`:

```bash
python experiments/run_experiment.py approach=oracle environment=pr2packed_generalized \
    eval_seed="$EVAL_SEED"
```

`approach=oracle` dispatches on the environment choice name
(`robocode.approaches.oracle_approach.ORACLE_TARGETS`), so it also runs the existing
kinder oracles (`obstruction2d_medium`, `clutteredstorage2d_medium`,
`stickbutton2d_medium`, `pushpullhook2d`). An oracle is a solvability check and a
reference row to compare synthesized approaches against, not a scored method.

`robocode.oracles.pr2packed` is a reference TAMP policy: for each block it
samples a base pose that can reach both the block and a free plate cell (inverse
reachability), solves arm IK there, plans collision-free joint paths, and tracks them
with the environment's bounded delta actions, resampling a block's plan when a stage
fails. It exists to confirm the task is solvable — it solves 60/60 episodes across
3/4/5 blocks (means of 136/185/237 steps), comfortably inside `max_steps_for_count`.
`tests/oracles/pr2packed/` is that check.

Every candidate configuration and path is validated against the environment's own
collision test rather than the planner's, because ss-pybullet configures the two
separately and a path the planner accepts is not automatically one the environment
will execute.

Geometry, grasping, and collision checking come from
[ss-pybullet](https://github.com/caelan/ss-pybullet), vendored as a submodule under
`third-party/ss-pybullet/` along with its own nested
[motion-planners](https://github.com/caelan/motion-planners) submodule. Neither ships
packaging metadata, so they are imported off `sys.path` by
`robocode/environments/ss_pybullet.py` rather than installed by `uv`; `install.sh` already
runs `git submodule update --init --recursive`, which is all the setup they need. Nothing
here requires a FastDownward build, because nothing here runs the PDDLStream planner.

### LIBERO-PRO (manipulation benchmark, optional extra)

[LIBERO-PRO](https://github.com/uynitsuj/LIBERO-PRO) is a Franka tabletop manipulation benchmark (~80 task suites covering goal / spatial / object / 10-task mixes plus OOD and perturbation variants) built on MuJoCo via robosuite. It is vendored as a submodule under `third-party/LIBERO-PRO/` and gated behind the optional `libero` extra — it is **not** installed by default because it pins old upstreams (`robosuite==1.4.0`, `gym==0.25.2`, `robomimic==0.2.0`, `bddl==1.0.1`) and drags in a CUDA-enabled torch.

`install.sh` passes `--no-extra libero`, so the default install skips it entirely.

**Linux only.**
The extra cannot be installed on macOS: `robomimic 0.2.0` depends on `egl-probe`, which compiles an EGL loader, and EGL has no macOS implementation.
Use the Docker sandbox to run LIBERO from a Mac.

Install (into the same venv as the rest of robocode):

```bash
sudo apt-get install -y libegl1 libgl1 cmake  # EGL/GL runtime for MuJoCo; cmake builds egl-probe
uv sync --all-extras --dev                    # ~60 extra Python packages, several GB
```

First use of the `libero` package runs an interactive `input()` prompt asking where to store datasets; the test harness writes `~/.libero/config.yaml` automatically. If you hit the prompt manually, answer `N` — the default paths are fine for env rollouts (pre-recorded demos are not required).

List available benchmark suites:

```python
from libero import benchmark
print(list(benchmark.get_benchmark_dict().keys()))  # ~80 suites
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

## Sandbox

The agent runs inside a Docker container (`robocode-sandbox`) that provides full filesystem isolation, a restricted network, and a pre-built Python environment.

### Security model

| Layer | Mechanism |
|---|---|
| Filesystem | Docker bind-mount: agent can only write to `/sandbox` (the run's output dir) |
| Network | `init-firewall.sh` whitelists API endpoints for the configured provider (Anthropic, OpenAI, Google, etc.), GitHub IPs, and telemetry; blocks everything else via iptables. Extra domains are passed via `ROBOCODE_FIREWALL_EXTRA_DOMAINS`. |
| Write hook | Claude backend: `PreToolUse` hook in `.claude/settings.json` double-checks Write/Edit paths stay inside `/sandbox`. OpenCode backend: `"permission": "allow"` in `opencode.json` (Docker provides the isolation). |

The Apptainer backend (`container_backend=apptainer`, for HPC clusters with no Docker daemon) keeps the same filesystem isolation but has **no network firewall**: unprivileged Apptainer cannot grant `CAP_NET_ADMIN`, so `init-firewall.sh` is skipped and generated code runs with unrestricted network egress. Use Docker where the iptables allowlist matters.

### What the agent sees

| Path | Contents |
|---|---|
| `/sandbox/` | Working directory — agent writes `approach.py`, test scripts, etc. here |
| `/sandbox/primitives/` | Source files from `src/robocode/primitives/` (read reference) |
| `/robocode/.venv/bin/python` | Python 3.11 with all robocode dependencies pre-installed |
| `/robocode/third-party/kindergarden/` | The kinder env package, bind-mounted read-only from the host submodule |

### Start docker

#### Mac OS

Simply open the Docker Desktop application.
Look for the status indicator in the bottom-left corner of the GUI; it should say "Docker Engine Running".

#### Linux
```
sudo systemctl start docker
sudo systemctl enable docker
```

### Building the image

Build once from the repo root (rebuild when `pyproject.toml` / `uv.lock` change; not needed for `third-party/kindergarden` code changes):

```bash
bash docker/build.sh
```

The strict blackbox ablation (`approach.blackbox_strict=true`) uses a separate
dependency-clean image instead; build it with `bash docker/build_strict_blackbox.sh`.

### Using the OS-level sandbox (legacy)

The original macOS Seatbelt / Linux bubblewrap sandbox is still available (`container_backend: local` in `agentic.yaml`) but has a known limitation: it restricts filesystem *writes* but allows *reads* of the entire host filesystem.

Red team the sandbox:
```bash
python integration_tests/red_team_sandbox.py           # OS-level
python integration_tests/red_team_sandbox.py --docker  # Docker
python integration_tests/red_team_sandbox.py --strict-blackbox  # strict blackbox
```

## Experiments

Set the private evaluation-suite seed in your shell before using any experiment
command or checked-in launcher. The prompt avoids saving it in shell history:

```bash
read -rsp "Private evaluation seed: " EVAL_SEED
echo
export EVAL_SEED
```

Run an experiment:
```bash
python experiments/run_experiment.py approach=random environment=small_maze replicate_seed=0 eval_seed="$EVAL_SEED"
```

Run a sweep over multiple seeds and environments:
```bash
python experiments/run_experiment.py -m replicate_seed=0,1,2 eval_seed="$EVAL_SEED" environment=small_maze,large_maze approach=random
```

Analyze results from one or more runs:
```bash
python experiments/analyze_results.py multirun/
```

Browse runs in the browser (metrics, per-episode GIFs, and the sandbox git history of the generated `approach.py`), then open http://localhost:8000. The history view charts replay solve rate and per-commit effort, and can replay the same failed seed across versions to show where it was fixed:
```bash
python -m experiments.results_viewer --root . --port 8000
```

The viewer can also read ZIP result archives recursively from a Google Drive
folder. The recommended backend is [rclone](https://rclone.org/drive/), which
provides browser login without requiring every collaborator to create a Google
Cloud project. Install rclone from its official downloads on Linux or with
Homebrew on macOS:

```bash
# Linux: https://rclone.org/downloads/
# macOS:
brew install rclone

rclone config
```

In `rclone config`, create a remote named `robocode-drive`, choose Google
Drive, leave the client ID and secret blank, choose read-only access, and allow
browser authentication. The resulting token stays in rclone's user config
outside the repository. Then launch:

```bash
python -m experiments.results_viewer --drive-folder "<Google Drive folder URL>"
```

Only `.zip` files are downloaded. Name each archive `<Experiment ID>.zip` so
the tracker ID, Drive result, and local cache directory match without manual
renaming. Archives are extracted under the user's cache directory, and the
viewer scans that local copy. The Refresh button checks Drive again, downloads
changed archives, removes archives deleted remotely, and rescans the cache.
Unchanged extracted archives are left in place, so GIFs rendered by the viewer
stay local and survive refreshes. The Drive folder URL and rclone configuration
are runtime configuration and must not be committed. Their locations can be
overridden with
`ROBOCODE_RESULTS_DRIVE_FOLDER`, `ROBOCODE_RESULTS_CACHE`,
`ROBOCODE_RCLONE_REMOTE`, and `RCLONE_CONFIG`. The same viewer command works on
Linux and macOS as long as `rclone` is on `PATH`.

Rclone currently warns that its shared Google Drive OAuth client is scheduled
for retirement during 2026. The blank-client-ID setup is therefore a convenient
prototype path, not a permanent team dependency. Configure a team-owned OAuth
client on the same rclone remote before Google disables the shared client; the
viewer command and archive layout do not change.

### Agentic approach

The `agentic` approach launches a coding agent during `train()`. The agent reads the environment source code, figures out the state/action space and dynamics, and writes a `GeneratedApproach` class that is used at evaluation time. The agent can also write and run test scripts against the real environment to verify its solution before committing.

By default the agent uses the Claude Code CLI backend and runs in the Docker sandbox (requires `bash docker/build.sh` once):

```bash
python experiments/run_experiment.py approach=agentic environment=motion2d_easy eval_seed="$EVAL_SEED"
```

Set `approach.blackbox=true` to hide the environment source and force the agent to discover the dynamics empirically through a host-side env server instead of reading code. Add `approach.blackbox_strict=true primitive_level=none` to also remove environment dependencies and non-render helper APIs: the generated program uses a generic Python environment with only the standard library, NumPy, and SciPy, can interact through `reset` and `step`, and may use the configured MCP tools to render states. The frozen program is checked against the same import allowlist before scoring. See [docs/blackbox.md](docs/blackbox.md) for the architecture.

To use a different backend/model, override the `approach/backend` config:

```bash
# GPT-5.4 via OpenCode
python experiments/run_experiment.py approach=agentic approach/backend=opencode_gpt54 eval_seed="$EVAL_SEED"

# Local Ollama model
python experiments/run_experiment.py approach=agentic approach/backend=opencode_qwen eval_seed="$EVAL_SEED"

# Or override individual fields
python experiments/run_experiment.py approach=agentic approach.backend.backend=opencode approach.backend.model=google/gemini-2.5-pro eval_seed="$EVAL_SEED"
```

Available backend presets: `claude_opus5` (default), `claude_sonnet5`, `claude_opus48`, `claude_sonnet46`, `claude_haiku45`, `claude_ollama_qwen`, `opencode_gpt54`, `opencode_gpt4omini`, `opencode_gpt5nano`, `opencode_qwen`.

The experiment runner rejects the legacy local sandbox for generated-code
methods because that sandbox permits host filesystem reads. Use Docker or
Apptainer so experimenter-only evaluation configuration is outside the
synthesis sandbox.

To skip re-generation and load a previously generated approach:
```bash
python experiments/run_experiment.py approach=agentic environment=small_maze \
    eval_seed="$EVAL_SEED" \
    approach.load_dir=outputs/2026-02-16/16-00-41
```

Parallel sweeps each get their own container (named `robocode-sandbox-<uuid>`), so multiple runs never interfere:
```bash
python experiments/run_experiment.py -m replicate_seed=0,1,2 eval_seed="$EVAL_SEED" environment=small_maze,large_maze approach=agentic
```

Use the [joblib launcher](https://hydra.cc/docs/plugins/joblib_launcher/) to run jobs in parallel locally:
```bash
python experiments/run_experiment.py -m \
    approach=agentic \
    approach.container_backend=docker \
    replicate_seed=42,24,424,444,222 \
    eval_seed="$EVAL_SEED" \
    primitive_level=none \
    environment=motion2d_easy,obstruction2d_easy,clutteredretrieval2d_easy,clutteredstorage2d_easy,stickbutton2d_easy,pushpullhook2d \
    'hydra.sweep.dir=multirun/2026-02-23/no_primitives_5d_s42_24_424_444_222' \
    'hydra.sweep.subdir=r${replicate_seed}/${hydra:runtime.choices.environment}' \
    hydra/launcher=joblib hydra.launcher.n_jobs=4
```

### Experiment tracker

Hydra defines executable choices, while small campaign files select the conditions
intended for one study. Keep exploratory or smoke-test campaigns local, and commit
only study definitions that should be shared. Generate a local CSV without running
experiments:

```bash
python -m experiments.tracker.generate \
    path/to/campaign.yaml \
    --eval-seed "$EVAL_SEED" --dry-run
python -m experiments.tracker.generate \
    path/to/campaign.yaml \
    --eval-seed "$EVAL_SEED" \
    --output experiments/generated/my_campaign.csv
```

Campaign files call the repeated runs `replicate_seeds`; they never contain the
private evaluation seed. The generator requires that fixed seed explicitly and places
it in the ignored local CSV and shared Sheet. Each condition becomes one row whose
Hydra command sweeps every replicate while holding the evaluation suite fixed.
The generated Experiment ID is passed into Hydra, recorded in every `results.json`,
and used as the exact parent directory under `multirun/`. Each invocation creates a
timestamped run beneath that parent with one `replicate_<replicate_seed>` directory
per replicate, so the condition folder can be uploaded to Drive without renaming it.
The ID fingerprints the complete Hydra-composed condition and both seed fields. Hydra
defaults such as access mode, model/backend, and the 60-second evaluation timeout are
materialized in every generated command. Editing any executable setting or seed
protocol therefore appends a distinct run instead of relabeling earlier results.
Study campaigns use the named `primitive_level=none|low_level|bilevel` config choices,
which resolve to the primitive list consumed by `build_primitives()`. Explicit
constraints exclude invalid cells such as `primitive_level=bilevel` with
`approach.blackbox=true`, and Hydra composition catches missing config choices.

Install the optional Google client and synchronize the generated CSV:

```bash
uv sync --extra tracker
python -m experiments.tracker.sync_google_sheet \
    experiments/generated/my_campaign.csv --sheet-id SPREADSHEET_ID
```

The sync uses Experiment ID as its key. It updates generated columns only for the exact
same canonical run, appends changed conditions or seed protocols, marks removed
conditions from the synchronized campaign inactive, and never writes existing Owner,
Status, Progress, Priority, Notes, Results, or Git SHA cells. It also rejects a
same-ID seed change as malformed input. New Sheets receive a native table with People,
file, and dropdown column types.
Generated categorical columns such as Campaign, Environment, Method, Primitive Level,
Access, Model / Backend, and Active are dropdown chips whose choices refresh from all
rows in the tracker, including inactive experiments from older campaigns. Priority is
placed immediately after Replicate Seeds and Evaluation Seed. Dropdown-chip colors can
be customized directly in the Google Sheets UI without changing the cells'
backgrounds; an unchanged sync preserves that native chip styling.
Status and Owner are the first two columns so the Sheet reads as a work queue at a
glance.

Authentication uses gspread's desktop OAuth flow. By default it reads
`~/.config/gspread/credentials.json` and stores the authorized-user token outside the
repository. Override those paths with `--credentials` / `--authorized-user` or the
`GOOGLE_OAUTH_CLIENT_SECRET` / `GOOGLE_AUTHORIZED_USER` environment variables. Never
commit either credential file.

The generated `approach.py` and full agent log are saved under `sandbox/` in the run's output directory (e.g. `outputs/2026-02-16/16-00-41/sandbox/`).

### Plain-LLM approaches

`llm_genplan` and `best_of_k` call a model directly (messages in, code out) with no tools and no agent loop. They take their model from the `approach/completion` config group, which is separate from the `approach/backend` group the agentic approach uses:

```bash
# Default: Opus 5 through the Claude CLI
python experiments/run_experiment.py approach=llm_genplan environment=small_maze eval_seed="$EVAL_SEED"

# Same approach on a different model
python experiments/run_experiment.py approach=best_of_k approach/completion=cli_sonnet5 eval_seed="$EVAL_SEED"
```

The `cli_*` presets drive the same authenticated Claude CLI as the agentic backend, so these runs need `claude auth login` (or `ANTHROPIC_API_KEY`) and no separate setup. The CLI applies an irreducible ~2k-token system prompt that cannot be stripped, so these baselines are prompted with that preamble present; the `anthropic_*` presets call the Messages API directly when a prompt with nothing else in it is required.

| Completion preset | Provider | Model |
|---|---|---|
| `cli_opus5` (default) | Claude CLI | `claude-opus-5` |
| `cli_sonnet5` | Claude CLI | `claude-sonnet-5` |
| `cli_opus48` | Claude CLI | `claude-opus-4-8` |
| `cli_sonnet46` | Claude CLI | `claude-sonnet-4-6` |
| `cli_claude` | Claude CLI | `sonnet` (alias; the CLI picks the generation) |
| `anthropic_opus` | Messages API | `claude-opus-4-8` |
| `anthropic_sonnet` | Messages API | `claude-sonnet-4-6` |
| `ollama_qwen` | OpenAI-compatible | `qwen3.6` (local Ollama) |
| `vllm` | OpenAI-compatible | `Qwen/Qwen3.6-35B-A3B` (local vLLM) |

The `anthropic_*` presets bill the Messages API and need `ANTHROPIC_API_KEY`; their `input_cost_per_mtok` / `output_cost_per_mtok` fields turn reported token usage into an estimated `cost_usd`, which bounds `approach.max_budget_usd`. The CLI reports its own cost, and the local presets report none.

### Planner baselines

Two per-instance planners solve each evaluation seed from scratch, with no LLM in the loop; they report `planning_time` and `plan_found` per episode and spend no budget. Both plan within the shared `eval_timeout`.

`bilevel_planning` runs kinder-baselines' SeSamE planner on every family for which `kinder_bilevel_planning` ships models: the kinematic-2D families, DynObstruction2D, DynPushPullHook2D, Transport3D, the kinematic Shelf3D, and Tossing3D (one cube only; other counts score as unsolved with a `planner_unsupported` flag). The family mapping is inferred from the env (`src/robocode/utils/bilevel.py`), so no env config needs a key. Some families need the planner settings their upstream configs use, passed as approach overrides:

```bash
python experiments/run_experiment.py approach=bilevel_planning primitive_level=none \
    environment=transport3d_generalized approach.max_skill_horizon=1000 eval_seed="$EVAL_SEED"
# dynpushpullhook2d: approach.samples_per_step=20 approach.max_abstract_plans=5
# shelf3d:           approach.max_skill_horizon=1000
# tossing3d:         approach.max_abstract_plans=1 approach.samples_per_step=5 approach.max_skill_horizon=400
```

`pddlstream_planning` runs the PDDLStream domain that kinder-baselines ships for Packing3D (`kinder-pddlstream-planning`) on a twin simulator and replays the plan against the evaluated environment in lockstep. It needs the `pddlstream` extra, whose install compiles FastDownward (`make` and a C++ compiler required):

```bash
uv sync --extra bilevel --extra pddlstream
python experiments/run_experiment.py approach=pddlstream_planning primitive_level=none \
    environment=packing3d_generalized eval_seed="$EVAL_SEED"
```

The pinned pddlstream fork bundles a FastDownward whose `search/ext/optional.hh` does not compile with GCC 13+ or recent clang (`tl::optional<T&>` carries a stale `emplace`; removing that member is the fix upstream FastDownward applied). Until the fork carries the fix, build it from a patched checkout and install it with `uv pip install --no-deps <path>`. PDDLStream writes its scratch files to `temp/` and `statistics/` in the working directory; both are ignored.

### Replicates and the evaluation suite

`replicate_seed` and `eval_seed` have deliberately different roles:

- `replicate_seed` identifies one independent replicate and seeds randomness
  controlled by Robocode, such as an approach's NumPy generator and action-space
  sampling. It does not seed Claude or make an agentic run reproducible.
- `eval_seed` is a fixed team value supplied explicitly in final-run commands.
  Robocode uses it to derive the same ordered evaluation episode suite for every
  method and replicate, so score differences are not caused by different sampled
  test suites. The checked-in default is `null`, and the runner fails if a command
  omits the value. Keep it out of public configs and agent-visible inputs.

The generalized synthesis agent does not receive `eval_seed` or the derived
episode seeds. Hydra's full configuration remains on the experimenter side, and
only the child `sandbox/` directory plus filtered source are mounted into Docker
or Apptainer. Per-instance methods are a separate protocol: they receive the one
derived episode seed they are solving, but not the master `eval_seed` used to
construct the suite.

For variable-object-count environments, evaluation sweeps the configured design
and held-out counts on that fixed episode schedule. `results.json` retains the
per-count curve and also reports `design_count_solve_rate` and
`held_out_count_solve_rate` separately.

The [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code/cli-usage)
and [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create)
do not expose a sampling-seed control. Consequently, replicates capture
uncontrolled model-generation variation even when `replicate_seed` is held
fixed; session IDs and model names are not random seeds.

This isolation boundary protects the synthesis process and its tool calls. The
generated policy is currently loaded by the trusted experiment runner for
rollout; it is reviewed as an experiment artifact rather than treated as
hostile code. Protecting the host from a deliberately malicious generated
policy would require running policy inference behind a separate process or
container boundary as well.

#### Example: small_maze

On `small_maze`, the agent independently discovered A* pathfinding and achieved a **100% solve rate with optimal path lengths** (mean 2.3 steps across 10 episodes):

```json
{
  "mean_eval_reward": -2.3,
  "mean_eval_steps": 2.3,
  "solve_rate": 1.0,
  "num_eval_tasks": 10
}
```

<details>
<summary>Generated <code>approach.py</code> (A* pathfinding)</summary>

```python
"""Optimal approach for MazeEnv using A* pathfinding algorithm."""

import heapq
from typing import Optional


class GeneratedApproach:
    """Optimal maze solver using A* pathfinding."""

    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.planned_path: Optional[list[tuple[int, int]]] = None
        self.path_index = 0

        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3

        self.action_to_delta = {
            self.UP: (-1, 0),
            self.DOWN: (1, 0),
            self.LEFT: (0, -1),
            self.RIGHT: (0, 1)
        }

    def reset(self, state, info):
        self.planned_path = self._astar_search(state)
        self.path_index = 0

    def get_action(self, state):
        if self.planned_path and self.path_index < len(self.planned_path) - 1:
            next_pos = self.planned_path[self.path_index + 1]
            dr = next_pos[0] - state.agent[0]
            dc = next_pos[1] - state.agent[1]
            for action, (delta_r, delta_c) in self.action_to_delta.items():
                if (dr, dc) == (delta_r, delta_c):
                    self.path_index += 1
                    return action
        return self._greedy_action(state)

    def _astar_search(self, state) -> Optional[list[tuple[int, int]]]:
        start, goal = state.agent, state.goal
        heap = [(self._heuristic(start, goal), start, 0, [start])]
        visited = set()
        while heap:
            _, current, g_score, path = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                return path
            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if (0 <= nr < state.height and 0 <= nc < state.width
                    and neighbor not in state.obstacles
                    and neighbor not in visited):
                    new_g = g_score + 1
                    heapq.heappush(heap, (
                        new_g + self._heuristic(neighbor, goal),
                        neighbor, new_g, path + [neighbor]
                    ))
        return None

    def _heuristic(self, pos1, pos2) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _greedy_action(self, state) -> int:
        dr = state.goal[0] - state.agent[0]
        dc = state.goal[1] - state.agent[1]
        if abs(dr) >= abs(dc):
            return self.DOWN if dr > 0 else self.UP
        return self.RIGHT if dc > 0 else self.LEFT
```
</details>

<details>
<summary>Agent log (19 turns, $0.36)</summary>

The agent's workflow:
1. Explored the sandbox directory and read `maze_env.py` and `base_env.py`
2. Analyzed the state type (`_MazeState` with `agent`, `goal`, `obstacles`, `height`, `width`), action space (4 directional moves), and reward structure (-1 per step)
3. Identified this as a shortest-path problem and implemented A* with Manhattan distance heuristic
4. Wrote and ran three test scripts verifying correctness on simple mazes, complex mazes with obstacles, impossible mazes (graceful fallback), and optimality of path lengths
5. Cleaned up test files and produced the final `approach.py`

```
18:32:05 Agent: I'll analyze the environment source files and create an optimal approach.py.
18:32:06 Tool call: Glob({'pattern': '**/*.py'})
18:32:08 Agent: Let me explore the main environment files to understand the structure:
18:32:09 Tool call: Read({'.../maze_env.py'})
18:32:09 Tool call: Read({'.../base_env.py'})
18:32:11 Agent: [Analyzed state/action space, identified A* as optimal strategy]
18:32:40 Tool call: Write({'.../approach.py'})
18:32:43 Agent: Now let me create a simple test script to verify the approach works:
18:33:00 Tool call: Write({'.../test_approach.py'})
18:33:03 Tool call: Bash({'command': 'python test_approach.py'})
18:33:07 Agent: Excellent! Testing on a more complex maze...
18:33:22 Tool call: Write({'.../test_complex_maze.py'})
18:33:26 Tool call: Bash({'command': 'python test_complex_maze.py'})
18:33:31 Agent: A* correctly found optimal 12-step path around all obstacles.
18:33:39 Tool call: Write({'.../test_real_state.py'})
18:33:54 Tool call: Bash({'command': 'python test_real_state.py'})
18:33:58 Agent: Works with actual _MazeState class.
18:34:09 Tool call: Write({'.../test_optimality.py'})
18:34:11 Tool call: Bash({'command': 'python test_optimality.py'})
18:34:13 Agent: All test cases produce optimal solutions.
18:34:14 Tool call: Bash({'command': 'rm test_*.py'})
18:34:25 Session done: turns=19, cost=$0.3628055, error=False
```
</details>
