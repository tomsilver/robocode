# ICRA 2027 experiment recovery

This file records the experiment commands running or queued on `agni` on
2026-09-12. Before restarting anything after a machine or shell failure, inspect
`multirun/` and `first_attempt_eval/` and remove already-completed replicate seeds
from the command. Do not launch a complete command blindly: Hydra creates a new
timestamped sweep rather than resuming completed replicates.

All campaign commands use the fixed evaluation seed `792075` and replicate seeds
`42,24,424,444,222`.

## Queue 1: Rearrange, then Sweep Into Drawer

```bash
TMPDIR="$PWD/.robocode-sandbox-mounts" uv run python experiments/run_experiment.py -m environment=rearrange3d approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=rearrange3d__llm_genplan__none__cli_opus5__timeout_60s__80185e71 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/rearrange3d__llm_genplan__none__cli_opus5__timeout_60s__80185e71/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}' && TMPDIR="$PWD/.robocode-sandbox-mounts" uv run python experiments/run_experiment.py -m environment=sweepintodrawer3d approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=sweepintodrawer3d__llm_genplan__none__cli_opus5__timeout_60s__8adf7e50 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/sweepintodrawer3d__llm_genplan__none__cli_opus5__timeout_60s__8adf7e50/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
```

## Queue 2: Sweep Simple, Tossing, then Balance Beam

```bash
TMPDIR="$PWD/.robocode-sandbox-mounts" uv run python experiments/run_experiment.py -m environment=sweepsimple3d_generalized approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=sweepsimple3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__7f110168 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/sweepsimple3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__7f110168/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}' && TMPDIR="$PWD/.robocode-sandbox-mounts" uv run python experiments/run_experiment.py -m environment=tossing3d_generalized approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=tossing3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__0c3e12c6 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/tossing3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__0c3e12c6/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}' && TMPDIR="$PWD/.robocode-sandbox-mounts" uv run python experiments/run_experiment.py -m environment=balancebeam3d approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=balancebeam3d__llm_genplan__none__cli_opus5__timeout_60s__6bf31b55 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/balancebeam3d__llm_genplan__none__cli_opus5__timeout_60s__6bf31b55/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
```

As of this note, Sweep Simple had completed and Tossing was running. Balance Beam
was still queued.

## Queue 3: PDDLStream-derived environments

Initialize the runtime dependency once:

```bash
git submodule update --init --recursive third-party/ss-pybullet
```

Then run the checked-in serial script:

```bash
./scripts/run_pddlstream_genplan_serial.sh
```

The script runs Packing, Blocked, and Rovers in that order and exits immediately
at an environment-level nonzero status. As of this note, Packing replicate 42 was
running.

## First-attempt reevaluation

```bash
TMPDIR="$PWD/.robocode-sandbox-mounts" uv run python scripts/evaluate_genplan_first_attempts.py multirun --jobs 12
```

The evaluator skips completed outputs, so this command is safe to rerun after a
failure. It was at 100/106 when this note was written.

## Static-analysis backup status

The final-policy analyzer, plotter, and tests are pushed on draft PR #214
(`codex/static-policy-complexity`). The history/evolution scripts and generated
CSV/PNG artifacts present in the main checkout on 2026-09-12 were untracked and
are not backed up by this branch.
