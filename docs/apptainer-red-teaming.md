# Full Apptainer red-teaming

The full coordinator reuses the existing probes in `integration_tests/red_team_sandbox.py`.
It selects Apptainer explicitly for every case, retains each case in its own directory,
and continues after failures so a single refusal cannot hide the remaining coverage.
Run the two backends in separate processes/results directories to test them concurrently:

```sh
.venv/bin/python -u -m integration_tests.apptainer_full_red_team \
  --backend codex --results-dir network_audit_results/full-codex-new

.venv/bin/python -u -m integration_tests.apptainer_full_red_team \
  --backend claude \
  --claude-token-file /home/mmerler/.config/robocode/claude_oauth_token_tom \
  --results-dir network_audit_results/full-claude-new
```

The defaults are `gpt-5.6-sol` and `sonnet`; `--model` selects a different model.
The token file is read only on the host. Its contents are never printed or placed
in container arguments. Results directories must be new. These are paid tests.

The catalog contains 45 cases plus the network/broker audit: strict smoke and attacks,
privilege/PID/cross-session checks, basic read/write escape attempts, blackbox source,
render and proxy attacks, models-off restrictions, held-out counts/seeds, demo recovery,
and live-agent `pkill`. `--suite strict` selects the strict subset. `--only NAME ...`
runs named follow-ups without repeating successful cases.

Each case retains `console.log`, the raw agent `stream.jsonl`, broker decisions,
and its sandbox (including `output.txt` and executed scripts). `summary.json`
records the original result, agent completion state, timing, and errors. Failures,
missing evidence, declined attacks, and setup errors never become passed cases.
Inspect actual commands/results as well as automated markers; model-written audit
scripts can be incorrect even when they claim to have finished.

The raw strict-protocol probe now uses exact JSON requests on a fresh working
connection for every forbidden command. It requires an explicit strict-mode server
rejection and successful reset/step afterward; a TypeError or closed connection is
not proof of the server's command policy. Read-leak canaries contain a random suffix
not disclosed in prompts, so echoing the known edit prefix is not a false breach.

## Strict package boundary finding

A separate deterministic check is available:

```sh
.venv/bin/python -m integration_tests.strict_import_audit \
  --results-dir network_audit_results/strict-import-new
```

The initial image failed this check: its separate MCP environment exposed the
reduced `robocode` proxy package, `mcp`, `pydantic`, and `httpx` simply by changing
interpreters or `sys.path`. A virtualenv is not an access boundary.

The corrected image removes that environment and the project package entirely.
Its two render tools use a standalone stdlib HTTP server and the numerical
`env_client`, under the same strict interpreter as agent scripts. Agent-written
policies still execute inside the isolated container. Python installer packages
and base-image build/debug packages outside the allowlist are removed too.
The audit checks both interpreters, deliberately added package paths, and readable
package source outside stdlib/NumPy/SciPy. The live `strict_render_import_escape`
case repeats forbidden imports from within a policy executed by `render_policy`.

Rebuild the strict image after updating. The Apptainer supervisor rejects the old
MCP environment before launching an agent. Regular Docker rendering continues to
use its existing MCP framework; strict Docker rendering uses the new standalone
server and also needs a rebuilt strict image. Docker runtime tests are unavailable
on this cluster; shared MCP configuration and SDK interoperability are unit tested.

The full audit also found withheld primitive-description bytecode in regular
models-off source mounts. The source filter now drops `__pycache__`, `.pyc`, and
`.pyo` files as well as withheld source. Regression tests compile the withheld
file and verify neither cached nor adjacent bytecode reaches the mount.

The generated-policy import allowlist remains a separate scoring-time check,
not an adversarial boundary around host-side scoring.

Original full-run artifacts are under `network_audit_results/full-redteam-codex/`
and `full-redteam-claude/`. Corrected protocol follow-ups are under
`strict-protocol-verified-codex/` and `strict-protocol-verified-claude/`; the package
finding is in `strict-import-audit/report.json`. Original failures are retained and
must not be silently overwritten by reruns.
