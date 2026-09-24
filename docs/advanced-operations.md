# Advanced operations

These utilities support experiment management and inspection. For the current paper protocol and local evaluation commands, start with the [README](../README.md). Run commands from the repository root after activating `.venv` and setting `EVAL_SEED` as described there.

## Results viewer and Google Drive

The viewer can also read ZIP result archives recursively from a Google Drive
folder. The recommended backend is [rclone](https://rclone.org/drive/), which
supports OAuth browser login; a shared team OAuth client can avoid each collaborator
creating a separate Google Cloud project. Install rclone from its official downloads on Linux or with
Homebrew on macOS:

```bash
# Linux: https://rclone.org/downloads/
# macOS:
brew install rclone

rclone config
```

In `rclone config`, create a remote named `robocode-drive`, choose Google
Drive, configure OAuth according to rclone's current Drive instructions, choose read-only access, and allow
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

## Experiment tracker

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


## History and rendering

`record_approach_history=true` uses [approach_history.py](../src/robocode/utils/approach_history.py) to replay commits containing `approach.py` from the sandbox Git repository. It renders one episode per snapshot using the replicate seed and its helper's step budget; it does not run each snapshot through the paper's common 100-instance suite. For a comparable score, recover the complete desired snapshot (including siblings) into a separate `sandbox/` and use the README's `approach.load_dir` workflow.

The viewer can render selected episodes and inspect saved history. Keep original generation results separate from reevaluation results so generation cost/provenance is not overwritten.

## Older LLMGenPlan submissions

Current LLMGenPlan writes `impl0_candidate.py` before validation. Older archives may only have the response text. This uses the same parser as synthesis without querying a model; run it from the repository root after setting `GENPLAN_RUN` and `ONESHOT_RUN` as in the [README](../README.md#one-shot-from-llmgenplan):

```bash
export GENPLAN_RUN ONESHOT_RUN
python - <<'PYCODE'
import os
from pathlib import Path
from robocode.approaches.llm_genplan_approach import _parse_python_code

source = Path(os.environ["GENPLAN_RUN"]) / "sandbox" / "impl0_response.txt"
target = Path(os.environ["ONESHOT_RUN"]) / "sandbox" / "approach.py"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(_parse_python_code(source.read_text()))
PYCODE
```

Then run the One-shot evaluation command in the README. Record the source response and parser revision. Missing responses cannot be reconstructed from a later `approach.py`; invalid responses are failures, not grounds to pick another generation.

## Protocol diagnostics

[Integration scripts](../integration_tests) cover budgets, timeout handling, firewall launch modes, environment-server concurrency, and protocol latency. Inspect their `--help` and prerequisites before running them: several start containers or make paid model calls. Current Apptainer checks and the distinction between deterministic controls and paid agent probes are documented in the [isolation check guide](apptainer-red-teaming.md).
