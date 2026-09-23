# Coding Agents for Generalized Task and Motion Planning Problems

[Project website and videos](https://agenticgentamp.github.io/) · [Paper](https://agenticgentamp.github.io/assets/paper.pdf) · [Code](https://github.com/tomsilver/robocode)

RoboCode is the experiment code for **AgenticGenPlan**, our study of coding agents for generalized task and motion planning (TAMP). Given a task description and simulator access, an agent probes the environment, writes and tests Python, and returns one program that solves new instances of the task. We freeze that program and evaluate it without further LLM calls.

The main setting is **strict black box**: no environment source, injected primitives, or robotics libraries. Programs use Python's standard library, NumPy, and SciPy; the agent interacts through `reset`/`step` and render tools. Source access is an additional condition. This is a study of off-the-shelf coding agents, with no prescribed symbolic representation or planning algorithm.

The current manuscript compares Claude Code with Opus 5 and Codex with GPT-5.6 Sol and GPT-6 Astra across **28 environments** (25 KinDER, three PDDLStream). Their mean success rates are **74%, 51%, and 86%**, respectively, versus 28% for LLMGenPlan across all environments. Hand-engineered planners average 47% on the 16 environments they cover; the agents average 82%, 56%, and 95% on that same subset. Synthesized programs exploit task regularities to reduce computation as object counts grow, and logs show calibration, edge-case testing, and strategy refinement. Hard dynamic 3D tasks remain challenging. The linked public PDF may lag the manuscript's latest Astra additions.

## Installation

Use **Python 3.11**, pinned in [.python-version](.python-version) and used by CI and the container images. Run the commands below from the repository root in Bash.

### Host dependencies

| Dependency | When needed |
| --- | --- |
| Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) | All installations; uv can provision Python 3.11 |
| Docker with a running daemon | The Docker synthesis commands below; the current user must be able to run `docker` |
| Apptainer and Podman | Alternative to Docker for Claude/Codex agentic synthesis; Podman builds the SIF images |
| C/C++ toolchain, `make`, CMake | Planner dependencies, including the FastDownward build |
| EGL/OpenGL runtime libraries | Headless MuJoCo/PyBullet environments and rendering; on Debian/Ubuntu install `libegl1 libgl1` |
| Node.js/npm | Host CLI installation commands below; not needed when using only an API key and container-provided CLIs |
| Host Claude or Codex CLI | Login-based authentication for that backend; agent CLIs are also installed inside the images |

For a Debian/Ubuntu host, install the build tools and libraries used by CI with `sudo apt-get install build-essential cmake liblapack-dev libblas-dev libegl1 libgl1 libglu1-mesa`. Install uv and the selected container runtime separately.

```bash
git clone https://github.com/tomsilver/robocode.git
cd robocode
git submodule update --init --recursive
uv sync --dev
source .venv/bin/activate
```

This installs the core environment/agent stack. Optional extras in [pyproject.toml](pyproject.toml) are `bilevel`, `pddlstream`, `tracker`, `develop`, and `libero`. The convenience command `bash install.sh` initializes submodules and installs **all extras except `libero`**, including planners, tracker, and development tools; it therefore needs the planner build prerequisites. LIBERO-PRO's Python stack is opt-in even though its submodule is initialized. The stock comparison-only PDDLStream submodule has `update = none` and requires explicit initialization when used.

### Authentication and images

For login-based authentication, install the selected CLI on the host (the Codex version below matches the container build):

```bash
# Choose the backend(s) you will use:
npm install -g @anthropic-ai/claude-code
npm install -g @openai/codex@0.153.4
```

For Claude, authenticate the [Claude Code CLI](https://code.claude.com/docs/en/setup) with `claude auth login`, or supply `ANTHROPIC_API_KEY`. For Codex, use the [Codex CLI](https://developers.openai.com/codex/cli/) and run `codex login`, or supply `CODEX_API_KEY`. Use an account authorized for the selected model. Subscription access and API billing differ; the experiment budget is tracked model usage, not a promise of free usage or an exact invoice amount.

Docker forwards the selected backend's authentication; its Codex login-file path copies only `auth.json` into a throwaway home. Apptainer keeps real credentials in the host inference broker. Neither Codex path mounts host skills, instructions, configuration, or past sessions. See the [Apptainer guide](docs/apptainer-network-isolation.md) for its namespace requirements and supported providers.

```bash
# Main paper setting:
bash docker/build_strict_blackbox.sh
# Source-access conditions and LLMGenPlan:
bash docker/build.sh
```

For Apptainer agentic runs, build the corresponding SIF with `bash docker/build_strict_blackbox_sif.sh` or `bash docker/build_sif.sh`, and use `approach.container_backend=apptainer`. The build scripts create large temporary OCI archives; set `TMPDIR` to a disk-backed location if `/tmp` is RAM-backed. LLMGenPlan/Best-of-K and OpenCode do not support the isolated Apptainer transport.

Rebuild regular images when their Dockerfile or dependency inputs change. Rebuild strict images when the strict Dockerfile, entrypoint, or copied render/client sources change. Updating an image does not update launcher code in a different checkout.

### Planner setup

```bash
uv sync --extra bilevel --extra pddlstream --dev
```

`bilevel` installs KinDER's SeSamE models/planner. `pddlstream` installs KinDER's Packing3D domain and the packaged PDDLStream fork, whose installation builds FastDownward. The PR2 Packing/Blocked and Rovers adapters also use the vendored `third-party/ss-pybullet` stack. Agentic runs do not need these optional planner extras.

The pinned PDDLStream revision can fail to compile with GCC 13+ or recent Clang because of its bundled `tl::optional<T&>::emplace`. If encountered, use a compatible toolchain or a separately patched checkout installed with `uv pip install --no-deps /path/to/patched-pddlstream`; record that patch and compiler with your results. Keep the compiled dependency and its provenance with the run environment.

## Quickstart

Choose your own nonnegative integer evaluation-suite seed, keep it fixed across methods and replicates, and keep it outside agent-visible inputs. The prompt avoids putting it in shell history:

```bash
read -rsp "Evaluation-suite seed (nonnegative integer): " EVAL_SEED
printf '\n'
export EVAL_SEED

python experiments/run_experiment.py \
  approach=agentic approach/backend=claude_opus5 \
  environment=motion2d_generalized primitive_level=none \
  approach.container_backend=docker \
  approach.blackbox=true approach.blackbox_strict=true \
  approach.max_budget_usd=20.0 \
  replicate_seed=42 eval_seed="$EVAL_SEED" \
  num_eval_tasks=100 eval_timeout=60 \
  hydra.run.dir=outputs/paper-motion-opus-r42
```

This synthesizes **one generalized policy**, then evaluates it on 100 instances with a 60-second episode timeout. It can consume $20 in model usage. To inspect the composed configuration without synthesis or evaluation, append `--cfg job --resolve`. The paper's strict setting must be explicit: the software defaults still select `approach=random`, `environment=small_maze`, and `primitive_level=low_level`; `approach=agentic` alone uses source access.

## Paper experiments

### Environment map

Each row is one paper environment, not one object count. Use the linked config's filename without `.yaml` as `environment=...`. A dash means there is no planner baseline for that paper environment. SeSamE uses `approach=bilevel_planning`; PDDLStream uses `approach=pddlstream_planning`.

| Family | Paper environment | Hydra environment config | Planner |
| --- | --- | --- | --- |
| Kinematic 2D | StickButton | [stickbutton2d_generalized](experiments/conf/environment/stickbutton2d_generalized.yaml) | SeSamE |
| Kinematic 2D | Obstruction | [obstruction2d_generalized](experiments/conf/environment/obstruction2d_generalized.yaml) | SeSamE |
| Kinematic 2D | ClutteredStorage | [clutteredstorage2d_generalized](experiments/conf/environment/clutteredstorage2d_generalized.yaml) | SeSamE |
| Kinematic 2D | ClutteredRetrieval | [clutteredretrieval2d_generalized](experiments/conf/environment/clutteredretrieval2d_generalized.yaml) | SeSamE |
| Kinematic 2D | Motion | [motion2d_generalized](experiments/conf/environment/motion2d_generalized.yaml) | SeSamE |
| Kinematic 2D | PushPullHook | [pushpullhook2d](experiments/conf/environment/pushpullhook2d.yaml) | – |
| Dynamic 2D | Obstruction | [dynobstruction2d_generalized](experiments/conf/environment/dynobstruction2d_generalized.yaml) | SeSamE |
| Dynamic 2D | PushPullHook | [dynpushpullhook2d_generalized](experiments/conf/environment/dynpushpullhook2d_generalized.yaml) | SeSamE |
| Dynamic 2D | PushT | [dynpusht2d](experiments/conf/environment/dynpusht2d.yaml) | – |
| Dynamic 2D | ScoopPour | [dynscooppour2d_generalized](experiments/conf/environment/dynscooppour2d_generalized.yaml) | – |
| Kinematic 3D | Obstruction | [obstruction3d_generalized](experiments/conf/environment/obstruction3d_generalized.yaml) | – |
| Kinematic 3D | Packing | [packing3d_generalized](experiments/conf/environment/packing3d_generalized.yaml) | PDDLStream |
| Kinematic 3D | Transport | [transport3d_generalized](experiments/conf/environment/transport3d_generalized.yaml) | SeSamE |
| Kinematic 3D | Table | [table3d_generalized](experiments/conf/environment/table3d_generalized.yaml) | – |
| Kinematic 3D | BaseMotion | [basemotion3d](experiments/conf/environment/basemotion3d.yaml) | SeSamE |
| Dynamic 3D | BalanceBeam | [balancebeam3d](experiments/conf/environment/balancebeam3d.yaml) | – |
| Dynamic 3D | ConstrainedCupboard | [constrainedcupboard3d_generalized](experiments/conf/environment/constrainedcupboard3d_generalized.yaml) | – |
| Dynamic 3D | Dynamo | [dynamo3d_generalized](experiments/conf/environment/dynamo3d_generalized.yaml) | – |
| Dynamic 3D | Rearrange | [rearrange3d](experiments/conf/environment/rearrange3d.yaml) | – |
| Dynamic 3D | ScoopPour | [scooppour3d_generalized](experiments/conf/environment/scooppour3d_generalized.yaml) | – |
| Dynamic 3D | Shelf | [dynamicshelf3d_generalized](experiments/conf/environment/dynamicshelf3d_generalized.yaml) | SeSamE |
| Dynamic 3D | SortClutteredBlocks | [sortclutteredblocks3d_generalized](experiments/conf/environment/sortclutteredblocks3d_generalized.yaml) | – |
| Dynamic 3D | SweepIntoDrawer | [sweepintodrawer3d](experiments/conf/environment/sweepintodrawer3d.yaml) | SeSamE |
| Dynamic 3D | SweepSimple | [sweepsimple3d_generalized](experiments/conf/environment/sweepsimple3d_generalized.yaml) | – |
| Dynamic 3D | Tossing | [tossing3d_generalized](experiments/conf/environment/tossing3d_generalized.yaml) | SeSamE |
| PDDLStream | Packing | [pr2packed_generalized](experiments/conf/environment/pr2packed_generalized.yaml) | PDDLStream |
| PDDLStream | Blocked | [pr2blocked_generalized](experiments/conf/environment/pr2blocked_generalized.yaml) | PDDLStream |
| PDDLStream | Rovers | [rovers_generalized](experiments/conf/environment/rovers_generalized.yaml) | PDDLStream |

The paper's Dynamic3D Shelf is `dynamicshelf3d_generalized`; `shelf3d_generalized` is a separate, preliminary **kinematic** Shelf environment. Configured `eval_counts` are cycled across the 100 evaluation episodes, and count-dependent step limits come from the environment wrapper. Fixed-size environments use the top-level `max_steps`.

### Methods and settings

Start from the quickstart's common protocol (`primitive_level=none`, $20, 100 instances, 60 seconds) and change the method/environment as follows:

| Paper condition | Approach and model selection | Access / scope |
| --- | --- | --- |
| Claude Opus | `approach=agentic approach/backend=claude_opus5` | Strict black box, all 28 |
| Codex Sol | `approach=agentic approach/backend=codex_gpt56sol` | Strict black box, all 28; medium reasoning |
| Codex Astra | `approach=agentic approach/backend=codex_gpt6` | Strict black box, all 28; high reasoning |
| Claude + source | `approach=agentic approach/backend=claude_opus5` | `approach.blackbox=false approach.blackbox_strict=false`, all 28 |
| Astra + source | `approach=agentic approach/backend=codex_gpt6` | Both black-box flags false; **only** `sweepsimple3d_generalized` and `scooppour3d_generalized` |
| LLMGenPlan | `approach=llm_genplan approach/completion=cli_opus5` | Full environment source, fixed validation/feedback loop, no agent tools |
| One-shot | First LLMGenPlan program | Same source and initial prompting; no refinement |
| Planner | `approach=bilevel_planning` or `approach=pddlstream_planning` | 16 supported environments above; no LLM usage |

The manuscript labels Opus agentic runs as high effort. The checked-in Claude agent backend delegates effort to the installed CLI; it does not pin an effort flag. Sol and Astra presets explicitly set `reasoning_effort: medium` and `high`. Preserve CLI versions and resolved settings when comparing runs. LLMGenPlan uses the separate **completion** config group; `cli_opus5` currently sets `max_thinking_tokens: 0`.

For example, the two supplementary Astra + source environments can be run as:

```bash
python experiments/run_experiment.py -m \
  approach=agentic approach/backend=codex_gpt6 \
  environment=sweepsimple3d_generalized,scooppour3d_generalized \
  primitive_level=none approach.container_backend=docker \
  approach.blackbox=false approach.blackbox_strict=false \
  approach.max_budget_usd=20.0 replicate_seed=42,24,424,444,222 \
  eval_seed="$EVAL_SEED" num_eval_tasks=100 eval_timeout=60
```

These are supplementary source-access runs, not an additional full 28-environment Astra condition. The manuscript reports Astra improving from 3% to 44% on SweepSimple and from 14% to 42% on ScoopPour3D with source access.

LLMGenPlan's checked-in default stops after four debug attempts or $20, whichever comes first. For a budget-limited run, remove that step cap explicitly:

```bash
python experiments/run_experiment.py \
  approach=llm_genplan approach/completion=cli_opus5 \
  environment=motion2d_generalized primitive_level=none \
  approach.container_backend=docker approach.max_budget_usd=20.0 \
  approach.max_debug_attempts=null approach.chain_of_thought=true \
  replicate_seed=42 eval_seed="$EVAL_SEED" num_eval_tasks=100 eval_timeout=60 \
  hydra.run.dir=outputs/paper-motion-genplan-r42
```

The loop can stop early when its training tasks are solved. Current code selects the best validated candidate for `sandbox/approach.py`, retaining the final candidate separately. Cost is checked between generations, so $20 is a stopping threshold rather than an exact bill. Use archived settings and submissions for historical comparisons.

### Planner commands

```bash
python experiments/run_experiment.py \
  approach=bilevel_planning environment=transport3d_generalized \
  primitive_level=none approach.max_skill_horizon=1000 \
  replicate_seed=42 eval_seed="$EVAL_SEED" num_eval_tasks=100 eval_timeout=60

python experiments/run_experiment.py \
  approach=pddlstream_planning environment=pr2packed_generalized \
  primitive_level=none replicate_seed=42 eval_seed="$EVAL_SEED" \
  num_eval_tasks=100 eval_timeout=60
```

PDDLStream also supports `packing3d_generalized`, `pr2blocked_generalized`, and `rovers_generalized`. Current SeSamE defaults are 10 abstract plans, 10 samples per step, and a 100-step skill horizon. Useful upstream planner overrides retained from the experiment setup are:

| Environment | Overrides |
| --- | --- |
| `dynpushpullhook2d_generalized` | `approach.samples_per_step=20 approach.max_abstract_plans=5` |
| `transport3d_generalized`, `shelf3d_generalized` (preliminary) | `approach.max_skill_horizon=1000` |
| `tossing3d_generalized` | `approach.max_abstract_plans=1 approach.samples_per_step=5 approach.max_skill_horizon=400` |

Consult each archived run's `.hydra/config.yaml` for its exact planner settings. The pinned Tossing3D planner supports **one cube only**, while the checked-in generalized config evaluates counts 1 and 2. Count 2 is scored unsolved with `planner_unsupported`; the current config therefore cannot recover the paper's reported Tossing planner score without its historical model revision/settings. The wrapper uses the same flag for any count whose model raises `NotImplementedError`. Keep those episodes in the denominator and recover the original configuration before claiming historical reproduction.

## Running and evaluating

### Five replicates and evaluation protocol

```bash
python experiments/run_experiment.py -m \
  approach=agentic approach/backend=codex_gpt6 \
  environment=motion2d_generalized primitive_level=none \
  approach.container_backend=docker \
  approach.blackbox=true approach.blackbox_strict=true \
  approach.max_budget_usd=20.0 replicate_seed=42,24,424,444,222 \
  eval_seed="$EVAL_SEED" num_eval_tasks=100 eval_timeout=60 \
  hydra.sweep.dir=multirun/paper-motion-astra \
  'hydra.sweep.subdir=r${replicate_seed}'
```

This schedules five independent synthesis runs (up to $100 tracked usage). Hydra runs them sequentially by default. For parallel execution, append `hydra/launcher=joblib hydra.launcher.n_jobs=2`, allowing for a separate container and simulator load per job. Use a new output directory for each campaign.

`replicate_seed` seeds randomness controlled by RoboCode and identifies a replicate; it does **not** make remote model generation deterministic. A fixed model name, seed, or session ID cannot reproduce a generated program exactly. Codex tracks root-agent and subagent usage against the shared synthesis budget using the rates in its preset; those estimates do not capture every billing tier or context-length adjustment.

`eval_seed` seeds NumPy's derivation of the common ordered episode suite. Keep the master seed, environment config, count schedule, and episode count identical across methods. The paper uses five runs per method/environment, 100 evaluation instances per program sampled from the same initial-state distribution used during synthesis, and a 60-second per-instance timeout. Current software still distinguishes unpinned-reset and evaluation count pools, as detailed below; preserve historical overrides when reproducing that distribution. It samples evaluation seeds to make accidental synthesis reuse unlikely, then checks retrospectively that they were not used. New user-selected seeds reproduce the protocol, not the paper's exact historical instances.

**Current count behavior:** checked-in configs still contain `design_counts`. Unpinned `reset` calls sample those counts; `options={"object_count": k}` can explicitly request another valid count. `eval_counts` sets the evaluation schedule. The emitted `count_regimes`, `design_count_solve_rate`, and `held_out_count_solve_rate` describe that configured split, **not evidence that an agent never tried a count**. The paper asks for programs supporting all valid counts and verifies exposure from synthesis records. Preserve and inspect reset telemetry/logs when making unseen-instance or unseen-count claims; do not infer them from field names.

A fresh run on current `main` is not an exact historical rerun. Preserve the run's resolved Hydra configuration, source and submodule SHAs, dependency lockfile, container identity, CLI/model versions, prompts, frozen program with sibling modules, and original seed protocol. The repository does not supply a complete historical campaign/archive manifest or the paper's private evaluation seed. Environment fixes, policy loading, isolation, model serving, and candidate selection have evolved; report such changes when reevaluating old policies.

### Saved artifacts and inspection

Hydra writes runs beneath `outputs/` or `multirun/` unless overridden. Keep the complete run directory:

| Artifact | Contents |
| --- | --- |
| `.hydra/config.yaml`, `overrides.yaml`, `hydra.yaml` | Experiment configuration and Hydra choices |
| `sandbox/approach.py` and sibling files | Frozen policy and its local dependencies |
| `sandbox/agent_log.txt` | Agentic synthesis log; backend session/usage files may also be present |
| `sandbox/implN_candidate.py`, `implN_response.txt`, `implN_score.json` | LLMGenPlan submissions, responses, and validation results |
| `sandbox/best_score.json`, `final_score.json` | LLMGenPlan's selected-best and literal-final generation indices |
| `results.json` | Per-episode outcomes, solve rate, timeouts/crashes, count summaries, and available generation cost/usage |
| `env_description.md` | Environment description supplied to the approach |

```bash
python experiments/analyze_results.py multirun/
python -m experiments.results_viewer --root outputs/ --port 8000
```

Open `http://localhost:8000` for the viewer. `render_videos=true` saves rollout videos. `approach.telemetry=true` records synthesis environment interactions (source-access telemetry requires a registered environment class). `record_approach_history=true` replays sandbox Git snapshots for inspection; it is not the 100-instance paper evaluation of each historical program. See [advanced operations](docs/advanced-operations.md).

### Reevaluating a frozen program

`approach.load_dir` expects a **run directory containing `sandbox/approach.py`** and skips synthesis. Preserve sibling modules and match the original environment, primitives, access restrictions, and evaluation protocol. Write reevaluation to a new directory:

```bash
python experiments/run_experiment.py \
  approach=agentic approach/backend=claude_opus5 \
  environment=motion2d_generalized primitive_level=none \
  approach.container_backend=docker \
  approach.blackbox=true approach.blackbox_strict=true \
  approach.load_dir=outputs/paper-motion-opus-r42 \
  replicate_seed=42 eval_seed="$EVAL_SEED" num_eval_tasks=100 eval_timeout=60 \
  hydra.run.dir=outputs/paper-motion-opus-r42-reeval
```

For LLMGenPlan, use `approach=llm_genplan approach/completion=cli_opus5` with its original run's `load_dir`; do not pass the agentic black-box flags. Final policy scoring runs on the host. Strict import checks enforce the allowed dependencies but are not hostile-code containment.

### One-shot from LLMGenPlan

The paper's **One-shot is the first program generated by LLMGenPlan**, after its initial summary/strategy prompts when enabled. It is not the first coding-agent commit, Best-of-K, or a new synthesis run.

For current runs, copy the saved first submission into a separate loadable run:

```bash
GENPLAN_RUN=outputs/paper-motion-genplan-r42
ONESHOT_RUN=outputs/paper-motion-oneshot-r42
mkdir -p "$ONESHOT_RUN/sandbox"
cp "$GENPLAN_RUN/sandbox/impl0_candidate.py" "$ONESHOT_RUN/sandbox/approach.py"

python experiments/run_experiment.py \
  approach=llm_genplan approach/completion=cli_opus5 \
  environment=motion2d_generalized primitive_level=none \
  approach.container_backend=docker approach.load_dir="$ONESHOT_RUN" \
  replicate_seed=42 eval_seed="$EVAL_SEED" num_eval_tasks=100 eval_timeout=60 \
  hydra.run.dir=outputs/paper-motion-oneshot-r42-eval
```

Keep the parent run and `impl0_response.txt` as provenance. If an older run lacks `impl0_candidate.py`, parse its `impl0_response.txt` with the repository's `_parse_python_code` helper, as shown in [advanced operations](docs/advanced-operations.md#older-llmgenplan-submissions). Never substitute the selected best or last submission. An invalid first program remains a failed first program: the loader can fail before writing `results.json`, so preserve the error and account for it explicitly instead of dropping that replicate. Setting `approach.max_debug_attempts=0` launches a new first-attempt run; it does not recover the historical One-shot artifact.

## Further documentation

- [Black-box access and protocol](docs/blackbox.md)
- [Apptainer network isolation](docs/apptainer-network-isolation.md) and [red-teaming](docs/apptainer-red-teaming.md)
- [Advanced operations](docs/advanced-operations.md): tracker, Drive viewer, history, and older saved submissions
- [Preliminary experiments and alternative configurations](docs/preliminary-experiments.md)

## Preliminary experiments

The repository retains useful paths that are **not used in the current paper**: OpenCode, older/alternative Claude presets, local-model completion backends, CDL, per-instance agentic synthesis, Best-of-K, random and oracle approaches, maze environments, fixed-difficulty variants, kinematic Shelf, injected primitive variants, LIBERO-PRO, and the CaP-X submodule. They remain available for exploration; their presence is not evidence of paper evaluation or a fresh compatibility test against each external provider.

The [preliminary guide](docs/preliminary-experiments.md) inventories these configurations and preserves their setup instructions, examples, and existing limitations. Paper-analysis notebooks are a separate follow-up release.
