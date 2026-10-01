# Checkpoint re-evaluation

Scores every git commit of finished agentic runs on the held-out evaluation suite, so
solve rate can be plotted against synthesis progress. During synthesis the agent
commits to `sandbox/` inside each replicate directory. Each distinct sandbox tree that
contains `approach.py` is one checkpoint. A checkpoint is scored on the run's
100 held-out episodes, exactly as the final program was.

The work is embarrassingly parallel and CPU-only: 2D environments use pymunk, 3D
environments use MuJoCo or pybullet, and the policies are plain Python. No step uses a
GPU.

## Pieces

| Step | Command | Where |
| --- | --- | --- |
| Pack runs into a store | `python -m experiments.reeval.prepare` | laptop or any node |
| Score one checkpoint | `python -m experiments.reeval.evaluate` | anywhere |
| Plan and run shards | `python -m experiments.reeval.worker plan` / `run` | login node / SLURM |
| Submit an array | `experiments/reeval/slurm/submit.sh <site.env>` | login node |
| Merge and validate | `python -m experiments.reeval.collect` | anywhere |

### Store

`prepare` turns each replicate directory into `runs/<run_key>/` with:

- `repo.bundle`: the sandbox history as a single `git bundle` file;
- `run.json`: the run's environment config, approach flags, seeds, commit list, original
  per-episode outcomes, and an estimate of the eval cost.

A run takes two files, about 1 MB on average, so the store copies quickly to cluster
filesystems that are slow with many small files.

The environment config comes from the run's own `.hydra/config.yaml`, not from
`experiments/conf/environment/`. Some campaigns used object counts that no committed
config file has, and the episode suite depends on them.

```bash
python -m experiments.reeval.prepare --root <results dir> --store <store> \
    [--include replicate_paths.txt]
```

`--include` lists replicate directories relative to `--root`, one per line. Runs
without `sandbox/.git` are skipped and counted. Planner and one-shot baselines have no
sandbox history.

### One checkpoint

```bash
export REEVAL_EVAL_SEED=<team evaluation seed>
python -m experiments.reeval.evaluate --store <store> --out <out> \
    --run <run_key> --tree <tree sha>
```

This exports the tree into a scratch directory, then calls `run_experiment.py` with
`approach.load_dir=<scratch>`. That loads `sandbox/approach.py` and its sibling modules
and only evaluates them, without starting an agent or a container. The call uses the
run's `eval_timeout`, `max_steps`, `num_eval_tasks` and `replicate_seed`, plus the
evaluation master seed. The 100 episode seeds derive from that master seed, so every
checkpoint faces the same instances as the final program did.

The result is `<out>/<run_key>/<tree>.json` with these fields:

- `status`: `ok`, `failed` (for example, `approach.py` does not import yet) or
  `timeout`;
- `results`: the harness `results.json`, with `solve_rate`, `by_count`, and
  `per_episode` entries holding `solved`, `num_steps`, `policy_time_s` and
  `env_time_s`;
- `commits`: the commits that share this tree;
- `cpu_model`, `host`, `wall_seconds` and `harness_revision`;
- `log_tail`: the end of the eval log, kept when the eval did not succeed.

Records are written atomically, and existing records are skipped, so interrupted work
resumes.

The policy budget (60 s per episode) is wall-clock time. Run one eval per physical
core, with BLAS threads set to 1, which the image already does. Records note the CPU
model, because a slower core can turn a near-budget success into a timeout.

### Running at scale

`worker plan` lists the checkpoints that have no record yet and writes them to a
manifest. It splits them into shards, longest estimated eval first, using the original
eval time of each run. `worker run --shard i` evaluates shard `i` with `--jobs`
processes. `submit.sh` does both: it plans inside the image on the login node, then
submits one array task per shard. Rerunning `submit.sh` plans again from what is still
missing.

```bash
experiments/reeval/slurm/submit.sh site/leonardo.env --select final  # validation pass
experiments/reeval/slurm/submit.sh site/leonardo.env                 # every checkpoint
```

Other options: `--match <regex>` restricts the plan to some run keys. `--limit N` keeps
the N cheapest checkpoints, for smoke tests. `--retry-failed` reschedules failed
records. Set `REEVAL_DRY_RUN=1` to print the `sbatch` command without submitting it.

The site file holds the paths, account, partition, cores per task and the evaluation
seed. Copy `slurm/site/<cluster>.env.example` to `slurm/site/<cluster>.env`; files
ending in `.env` are git-ignored.

### Validation

```bash
python -m experiments.reeval.collect --store <store> --out <out> --dest <tables dir>
```

This writes two tables:

- `checkpoints.csv`: one row per commit, with its index, time, message, status and
  solve rate;
- `validation.csv`: the final checkpoint compared with the run's original
  `results.json`, episode by episode.

Run the `--select final` pass first. All checkpoints are scored with one harness
revision, so runs whose original eval used an older harness can differ. Two examples
are the whole-rollout timeout before the per-policy budget, and older kinder revisions.
The validation table shows where.

## Image

```bash
bash docker/build_eval_sif.sh <kinder mimiclabs_scenes dir> [robocode-eval.sif]
```

The image contains the locked dependencies without extras, plus robocode and kinder at
the current commit. It also contains kinder's MuJoCo scene assets: `meshes/` and
`textures/` are not in git and are otherwise downloaded at first use. Evaluation needs
no network. The image is built locally (Docker, then Apptainer) and copied to the
cluster as a single file. Containers run with `--cleanenv`, and the seed reaches them
through `APPTAINERENV_`/`SINGULARITYENV_` variables.

## Cluster notes

- **Leonardo (CINECA):**
  - Partition `dcgp_usr_prod`: 112 cores per node, billed per reserved core.
  - The account needs DCGP budget; check it with `saldo -b --dcgp`.
  - `$WORK` holds the image, the store and the outputs. `$TMPDIR` is node-local
    scratch.
  - Compute nodes have no internet: stage data from a login node or the datamover.
  - Use `dcgp_qos_dbg` (30 min) for smoke tests.
- **Della (Princeton):**
  - The QOS follows `--time`, and jobs of 61 min or less go to the two-job `test` QOS.
  - Pin one CPU type with `--constraint`, because the policy budget is wall-clock time.
  - Use `/scratch/gpfs` for the image, the store and the outputs.
