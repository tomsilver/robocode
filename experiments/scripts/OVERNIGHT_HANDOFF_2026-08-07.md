# Overnight campaign handoff (2026-08-07)

Instructions for the operating agent running tonight's queue on the cluster.
Your job is to set up, launch, babysit, and record. You are NOT running the
analysis, and you must not modify the experiments (see Hard rules).

## What is being run, and why (one paragraph each)

**Two-shelf strict/twin pair.** A controlled test of the non-monotonicity
hypothesis: two environment arms that are bit-identical in physics and differ
in one default tuple (per-block target shelves). In the strict arm a
pre-stored block starts inside its own goal shelf and provably must leave it
before the task can be completed (goal count dips 1 -> 0 on every solution);
in the twin arm the same block targets the opposite shelf and a monotone
solution exists. One synthesis run per arm on the same seed; the comparison
is paired. The arms live on separate git branches so each sandboxed agent
reads source whose defaults match the environment it samples.

**Repair probe.** A controlled repair test on a frozen program from the
08-05 campaign (an Opus 5 run that scores 1.00 on the standard suite and
0/77 on a certified wall-band family it never tested). Three conditions,
identical except for one file: the sandbox starts from that frozen program
plus no instance list / four mid-distribution instances / four wall-band
instances, with the same neutral prompt line. Whether the band examples
cause a repair (and whether the standard suite is retained) is the
measurement.

## Pinned versions

- `two-shelf-clutteredstorage`: the pushed tip **containing this file** --
  it must include `28da2f5` ("Automate the smoke gate and preflight
  checks"); anything older lacks the automated gate. Verify:
  `git merge-base --is-ancestor 28da2f5 HEAD && echo ok`.
- `two-shelf-twin`: exactly `efc8e84`.
- `repair-probe`: exactly `d113793`.
- Repair kit content hashes (sha256, first 16 hex chars) -- verify with
  `sha256sum experiments/repair_probe/*` in the repair checkout:
  `runB_approach.py 049cb882d0845b67`, `instances_band.json
  fda912abb00e3641`, `instances_mid.json 250ceace11c82514`. Band list is 4
  entries (seeds 357, 455, 505, 649), mid list is 4 entries (seeds 3, 5,
  8, 12), all `object_count` 3, in that order.

If any pin fails to verify: stop and report. Do not fetch "newer" versions.

## Setup

1. Fresh clone next to the existing one (do not disturb `/data01/mmerler/robocode`):

   ```bash
   git clone https://github.com/tomsilver/robocode robocode-two-shelf
   cd robocode-two-shelf
   git checkout two-shelf-clutteredstorage
   git submodule update --init third-party/kindergarden
   ln -s /data01/mmerler/robocode/robocode-sandbox.sif robocode-sandbox.sif
   ```

2. Activate the same project environment used for the 08-05 campaign
   (needs `python` with `hydra` importable; the launch scripts check this).

3. Verify agent-CLI auth exactly as it worked on 08-05 (same image, same
   HOME credential mounts). If auth fails in the smoke run, stop and report;
   do not improvise credential plumbing.

The queue script creates the other two checkouts as worktrees
(`<root>-twin` on `two-shelf-twin`, `<root>-repair` on `repair-probe`) and
verifies, before spending: each checkout's branch, its arm's default
pattern in the env module, the sif symlink, and the submodule checkout.

## Launch

Detached, so it survives the login session:

```bash
setsid nohup bash experiments/scripts/overnight_queue_2026-08-07.sh \
  > overnight_queue.log 2>&1 &
```

Queue order (strictly sequential between stages, timestamps in the log):

1. **Smoke**: strict arm, seed 7, `max_budget_usd=2`, 10 eval tasks. The
   script then enforces the gate itself and aborts the queue on any
   failure: `results.json` present, `environment=two_shelf_clutteredstorage2d`
   in the overrides, model resolved to `claude-opus-5` (not the `opus`
   alias, which the CLI maps to Opus 4.8), env card's count example is the
   non-runnable `design_counts=[<count list>]` placeholder, `stream.jsonl`
   non-empty. Your only manual duty at this point: confirm in
   `overnight_queue.log` that the gate line "smoke gate passed" appeared.
2. **Strict s42 + twin s42** in parallel (matched pair), `max_budget_usd=20`,
   `num_eval_tasks=100`.
3. **Repair probe none / mid / band at s42**, sequential, same budget.

Expected wall time from the 08-05 campaign: runs took 17-190 min, median
~1 h. Whole first pass: roughly 5-10 h.

## What counts as success, failure, and hung

Per run, read only `results.json` and the queue log:

- **Valid completion**: `results.json` parses, has a solve rate over 100
  eval tasks (10 for the smoke), and `gen_stop_reason` is `success` or
  `error_max_budget_usd` (a capped run is a valid, budget-limited result).
- **Infrastructure failure**: any other stop reason, a missing/unparsable
  `results.json`, or a crash in the log (auth error, missing sif, disk
  full, killed process). Policy: retry the identical command exactly once,
  into a fresh output dir with an `_retry1` suffix on the seed segment
  (append `hydra.run.dir=.../s42_retry1` to the same command the script
  ran, copied from the log). Never rerun into an existing output dir.
  Second failure: stop that campaign, let the rest of the queue proceed,
  report.
- **Pairing**: a strict/twin pair is valid when both arms reach valid
  completion, original or `_retry1`. If one arm fails twice, keep the
  surviving arm's run and record the pair as incomplete -- do not rerun
  the surviving arm.
- **Hung vs rate-limited**: the harness auto-resumes rate-limited CLI
  sessions, so a quiet run can be healthy. Objective check: if a run's
  `stream.jsonl` has not grown for 45+ minutes AND no apptainer/CLI
  process for that run is alive, treat it as an infrastructure failure
  (above). If a process is alive, leave it alone.

## Record

When the queue finishes, write `outputs/first_pass_status_2026-08-07.md`
in the strict checkout: one line per run -- run dir (note the repair runs
live under the `-repair` sibling checkout's `outputs/`), start/end
timestamps from the queue log, `gen_stop_reason`, `agent_cost_usd`, solve
rate as numerator/denominator, retry status. Numbers only, no
interpretation. Keep `overnight_queue.log` and the manifest files the
scripts write under each checkout's `outputs/`.

## Second pass -- requires an explicit human go-ahead

Additional seeds are a human decision made after reading the first-pass
results; nothing here authorizes them. For reference only, the eventual
commands are one invocation per campaign with explicit seeds
(`two_shelf_arms_2026-08-07.sh 24 424`; `repair_probe_2026-08-07.sh 24`
from the repair checkout).

## Hard rules

- **Never modify the experiments.** No edits to `src/robocode/environments/`,
  `src/robocode/prompts.py`, approach code, the env or backend yamls, the
  instance lists, or the seed program. If a run fails for infrastructure
  reasons, follow the retry policy above -- never "fix" the experiment to
  make it pass.
- **Never write into a sandbox.** Nothing may be added to any run's
  `sandbox/` directory, and `approach.py` must not be opened for editing
  mid-run. The sandboxed agent must not be able to see anything you produce.
- **Do not read or summarize the agents' synthesis transcripts.** Analysis
  happens elsewhere, blinded to conditions. The only files you open are
  `results.json`, `.hydra/` configs, `env_description.md`, and the queue
  log.
- Git hygiene: plain follow-up commits only, no force-push, no PRs, no
  merges into main. Nothing in this job should need a commit at all.
- Spend: the queue as written is authorized (one $2 smoke + five $20-capped
  runs + at most one `_retry1` per failed run). Anything beyond that needs
  an explicit go-ahead.
