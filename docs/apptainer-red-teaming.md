# Apptainer isolation checks

Use the existing integration scripts after rebuilding the images. See
[network isolation](apptainer-network-isolation.md) for runtime requirements.

## Network and broker checks

```sh
python -m integration_tests.check_agent_internet_access \
  --container apptainer --backend codex --results-dir network_audit_results/network-codex
```

The Apptainer mode checks disconnected namespaces against reachable host-side
positive controls, then exercises the selected agent and the inference broker.
It covers direct sockets, DNS, host proxies, shell/network clients, package
installation, web tools, and attempts to bypass broker request validation.
Use `--backend claude` to exercise Claude, `--image-dir PATH` for images outside
the working directory, or `--no-agent` for deterministic checks without model calls.
A failed positive control is inconclusive, not evidence of isolation.

The same script retains its Docker webpage probe. `--probe webpage` explicitly
selects that narrower probe; it cannot establish a network isolation boundary.

## Full agent red-teaming

```sh
python -u -m integration_tests.red_team_sandbox --apptainer-full \
  --backend codex --results-dir network_audit_results/full-codex

python -u -m integration_tests.red_team_sandbox --apptainer-full \
  --backend claude --results-dir network_audit_results/full-claude
```

These commands run the existing attack catalogs using Apptainer and retain each
case separately. They continue after failures and include the network/broker
checks. Use separate processes and results directories for concurrent runs.
Results directories must be new. These tests make paid model calls.
`--model` overrides the backend default. `--claude-token-file PATH` optionally
loads Claude credentials on the host; credentials are not put in container arguments.

The suite covers filesystem escapes, process and session isolation, source and
package visibility, rendering, environment protocol restrictions, withheld models,
evaluation metadata, and demonstration recovery. `--suite strict` selects strict
blackbox coverage. `--only NAME ...` selects named cases for follow-up runs.
Existing Docker and individual red-team modes remain available; use `--help`.

Each case retains `console.log`, agent transcripts, broker decisions, and its
sandbox. `summary.json` records outcomes, completion state, timing, and errors.
Failures, declined attacks, missing evidence, and setup errors do not count as
passes. Inspect executed commands and results as well as automated verdicts.
Keep failed-run evidence and use a new directory for follow-ups.

## Strict package inventory

```sh
python -m integration_tests.red_team_sandbox --strict-import-audit \
  --results-dir network_audit_results/strict-imports
```

This deterministic check probes available interpreters, extra package paths, and
readable package sources outside the strict allowlist. The live
`strict_render_import_escape` case also probes imports from a rendered policy.
Strict images use a standalone rendering server under the same numerical Python
interpreter as agent scripts; a separate virtualenv is not an access boundary.
Rebuild strict images for both Apptainer and Docker when their definitions change.

These checks cover the agent sandbox. Host-side scoring is a separate boundary;
its policy import allowlist is not an adversarial isolation mechanism.
