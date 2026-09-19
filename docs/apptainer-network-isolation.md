# Apptainer isolation — implementation and audit, 2026-09-19

The updated launcher supports usable, unprivileged network isolation for **Codex
and Claude** on the tested cluster node `rk019192` (Apptainer 1.4.3, UID 1013).
This proves the agent-runtime boundary, not end-to-end experiment isolation:
`GeneratedProgramApproach._load_generated` still calls the host-side
`load_generated_approach`, which executes the generated policy during scoring.
That separate execution path is not isolated by the broker and must be addressed
before treating entire experiments as internet-blocked.

No Docker, sudo, or host firewall changes were needed. The previous launcher
shared host networking; disabling Codex web tools alone did not prevent fetching
websites or installing packages.

## Why there is a broker

A completely disconnected container cannot contact a model API. The agent now
runs with `--userns --net --network none`: its only network interface is loopback.
A small trusted program on the host, the **model broker**, accepts requests through
an explicitly mounted Unix socket. Unix sockets are local IPC and can cross this
boundary when their filesystem path is deliberately mounted.

The broker forwards only validated inference requests to a fixed OpenAI or
Anthropic HTTPS endpoint. It holds the real credentials outside the container.
It rejects arbitrary destinations, CONNECT, redirects, client-supplied routing
and authentication headers, provider-hosted web search, remote MCP, remote
image/file retrieval, and unsupported API operations. It accepts client-side tool
definitions so the agent can still run its local shell and experiment tools.
Its logs contain endpoint/status/rejection information, not prompts or tokens.

This is an explicit exception for model inference, not a general internet proxy.
An agent can submit permitted inference requests itself; the broker is not an
anti-abuse or spending boundary for that authorized API channel. Prompts and
inline raster images are intentionally sent to the model provider.

A **separate pinned relay** connects to one host experiment environment-server
port, selected by trusted launcher configuration. The agent can reset, step, and
render through the existing environment protocol. Editing its metadata cannot
select a different host service. This server remains part of the trusted surface.

## Code structure and Docker compatibility

There is one supported Apptainer transport, with separate responsibilities:

| Module | Responsibility |
| --- | --- |
| `apptainer_sandbox.py` | Agent launch; broker/relay lifetime; local CLI configuration |
| `model_broker.py` | Host credentials, fixed upstream selection, HTTP request policy |
| `isolated_transport.py` | Container namespace checks and fixed-destination byte relays |
| `apptainer_environment.py` | Trusted dependency preparation and clean child environment |

The relays do not implement another model policy; they deliver bytes to the broker
or one environment server. Provider hostname constants live in `backends/__init__.py`
and are shared with Docker's existing domain registry. The broker's `BrokerUpstream`
is a resolved, credential-bearing host connection, not a second provider registry.

The old Apptainer credential-forwarding helper, credential mounts, firewall-domain
arguments, and image-entrypoint dependency flags have been removed. There is no
legacy transport switch or fallback. Environment relays require an explicit trusted
`env_server_port`; `env_spaces.json` cannot select a host destination. GenPlan and
Best-of-K reject Apptainer at configuration time instead of exposing a dummy runner.

Docker keeps its existing credential mounts/environment, firewall domain settings,
and entrypoint execution. Its scripts' behavior is unchanged; only stale comments
about Apptainer were corrected. Docker's domain firewall and Apptainer's broker
provide different policies and are not selectable alternatives within Apptainer.
`ROBOCODE_FIREWALL_EXTRA_DOMAINS` remains a Docker setting, never a broker override.
Docker runtime tests require a collaborator's Docker-capable machine; unit tests
cover its command construction, auth, shared callers, and GenPlan dispatch here.

## Production behavior

- Every supported agent launch and resumed session uses the disconnected namespace.
  A supervisor checks non-root UID, loopback-only interfaces, no IPv4 routes,
  zero capabilities (including the bounding set), and `NoNewPrivs: 1` before
  starting the agent. Failure aborts; there is no host-network fallback.
- Filtered mounts, `--containall`, `--no-home`, `--cleanenv`, and PID isolation
  prevent default host-home mounts and inherited secrets. Real provider auth
  files are not mounted. Container API tokens are inert local placeholders.
- Regular Python environments are prepared and cached by a trusted installer
  **before** agent execution. That phase has network access but no agent files,
  sessions, or credentials. The completed environment is mounted read-only.
  The agent phase skips the image's online entrypoint, sets `UV_OFFLINE=1` and
  `PIP_NO_INDEX=1`, and is additionally blocked by the actual network namespace.
- Strict runs use the dependency-clean strict image. Codex web tools remain
  disabled by the pulled upstream configuration; broker enforcement also rejects
  attempts to enable hosted tools through raw API requests.
- OpenCode, custom upstreams, and the separate GenPlan/Best-of-K Apptainer path
  are currently unsupported and fail closed. They require a separate integration.
- Host credentials must be valid when a run starts; the broker does not implement
  OAuth refresh. Refresh/login happens on the trusted host.

Use this updated checkout. Old experiment worktrees retain their own old launcher
code and are **not** protected merely by reusing an updated SIF. The default
root-level SIFs were replaced with the tested Codex-capable images from
`outputs/codex-ready`; the previous images are retained with
`.pre-network-broker-20260919.bak` suffixes.

## Real test evidence

The audit pairs each isolated launch with an explicitly unrestricted **test-only
positive control**, using both regular and strict images. It checks namespace
identity, interfaces, routes, capabilities, and these actual communication paths:

- Python HTTPS and direct IPv4 TCP; curl, curl with a pinned IP, wget, Git HTTPS,
  Node HTTP, Bash `/dev/tcp`, and strict-image `pip download six==1.17.0`.
- DNS over UDP and TCP; controlled host IPv4/IPv6 loopback TCP, UDP, HTTP, and an
  explicit HTTP proxy; host abstract Unix sockets and an unmounted pathname socket.
- Raw sockets, route modification, namespace joining, nested namespaces, and a
  positive listener inside the container's own loopback namespace.

The live Codex and Claude runs execute the supplied probe unchanged through the
production launcher. Both passed: model calls worked while the tested forbidden
communication paths failed. All **14 broker attacks** per backend were rejected
with HTTP 403, testing both the loopback API endpoint and the mounted Unix socket:
CONNECT, GET, absolute URLs, hosted OpenAI/Claude web search, remote images, and
remote MCP. Prompts also ask agents to try their built-in web tools.

Further live checks passed:

- Codex and Claude resumed sessions, including absence of real credential files
  and secret environment variables, loopback-only networking, and blocked direct TCP.
- Claude strict environment reset/step plus MCP rendering to a PNG.
- Codex regular environment imports, and a real Motion2D reset/step rollout using
  the default regular SIF and the optional bilevel dependency environment.

**All successful Claude checks used the `_tom` token**, loaded on the host without
printing or mounting it. An initial Claude broker attempt returned HTTP 400 for
its context-management payload; the fixed, host-selected context-management beta
header resolved it and the complete live audit passed on rerun.

Artifacts are local and ignored by Git:

| Directory under `network_audit_results/` | Evidence |
| --- | --- |
| `cleanup-live-codex/` | Post-cleanup paired controls and live Codex broker attacks |
| `cleanup-live-claude/` | Post-cleanup paired controls and live Claude `_tom` broker attacks |
| `cleanup-strict-env-smoke/` | Post-cleanup pinned relay and MCP render |
| `cleanup-whitebox-rollout/` | Post-cleanup prepared bilevel environment and Motion2D rollout |
| `final-default-isolation/` | Final paired network controls using both default SIFs |
| `isolated-live-codex/` | Paired controls and successful live Codex attack suite |
| `isolated-live-v2-claude/` | Paired controls and successful live Claude attack suite |
| `strict-env-smoke/` | Strict environment and rendered PNG |
| `regular-smoke/` | Regular dependency imports |
| `default-whitebox-rollout/` | Default-image Motion2D rollout |
| `live-methods/` | Historical successful internet access through the old launcher |

Resume evidence is stored in the live run work directories. Summaries include
host/kernel/runtime and image fingerprints. Earlier artifacts call the unrestricted
control `production` and live results `live_baseline`; current code uses
`unrestricted_control` and `live_run` to avoid confusion.

External IPv6 had no working host-network control, so that particular test is
**inconclusive**, not a pass. Host IPv6 loopback isolation was positively tested.
The original strict image provided the positive `pip download` control and its
isolated download was blocked. The corrected strict image removes pip/setuptools
and the old MCP environment; both final images report pip as unavailable, not as
a successful blocked-download test. Missing reports, refused prompts, changed probe scripts,
failed model calls, and missing executables are never isolation successes.

## Reproduce

From the updated repository root, using fresh results directories:

```sh
.venv/bin/python -m integration_tests.red_team_sandbox \
  --network-isolation-only apptainer \
  --network-results-dir network_audit_results/new-audit
```

Add `--network-live-backend codex` or `--network-live-backend claude` for a paid
live agent audit (configured budget $2). Host credentials are required. Select a
specific pair of images with `--network-image-dir outputs/codex-ready` if needed.
The full `--apptainer-strict-blackbox` suite now includes its network and
package-install attacks as well as import, filesystem, and environment-protocol
attacks. The deterministic audit is required for strong network evidence; the
older webpage-only script reports a `BLOCKED` self-report as inconclusive (exit 2).

Run actual socket/container tests outside additional execution sandboxes that
forbid all sockets: such a sandbox can prevent Apptainer itself from starting and
would invalidate the test. The audit's in-namespace listener is a positive control
against this false pass.

## Code verification

The post-cleanup core regression run passed **259 tests**, with **19 Docker runtime
checks skipped** because Docker is unavailable. A further suite passed 60 checks
covering Best-of-K, retry routing, environment-server behavior, and provider domain
lists (the provider-list checks also appear in the core suite). Broker tests now
verify host-only credential loading instead of the removed credential-forwarding
helper. Namespace, request-policy, pinned-port, and secret-exclusion checks remain.

Mypy passed for eleven checked modules. Pylint and whitespace checks passed.

Docker's network launch, firewall, and credential behavior are unchanged.
The later strict-image fix switches strict Docker rendering to the same numerical
interpreter and standalone server, so strict Docker users must rebuild their image.
The shared source filter also now removes bytecode. Actual Docker execution still
needs verification on a Docker host.

## Scope of the conclusion

This cluster can run usable isolated Apptainer agents with this implementation.
A different cluster is not required by a fundamental rootless-Apptainer limit.
The trusted host, kernel, Apptainer, broker, dependency preparation, and environment
server remain part of the security boundary. Finite tests cannot prove the absence
of every kernel or protocol vulnerability. Repeat the audit on every execution
node/image and after runtime, broker, provider-protocol, or launcher changes.
Disabling tools alone, or switching container runtimes alone, is insufficient.

Primary references:

- [Apptainer 1.4 network virtualization](https://apptainer.org/docs/user/1.4/networking.html)
  documents the unprivileged `none` network.
- [OpenAI configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
  documents custom providers and transport settings.
- [Claude context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
  documents the context-management beta used by the fixed upstream headers.

## Full red-team follow-up and strict package correction

Both backends completed the full 44-case catalog plus the network/broker audit.
The original summaries include failures/inconclusives and are retained unchanged.
See `network_audit_results/full-redteam-review.md` for raw results and adjudication.

A real strict-image defect was found: changing interpreters or `sys.path` exposed
the reduced project/MCP packages. The rebuilt default image removes that entire
environment; both render tools use stdlib and the allowed numerical client.
The final import audit also inventories readable package sources and tests system
Python, the strict interpreter, and manually injected paths. Both agents tested
imports from inside a rendered policy. The supervisor refuses the original image.
A separate real bytecode exposure in models-off mounts was fixed by excluding
caches and compiled files. The generated-code host-scoring scope is unchanged.

Final artifacts: `final-network-codex/`, `final-network-claude/`,
`strict-import-final/`, and `strict-startup-guard/`, all under
`network_audit_results/`. Final network tests passed for both live agents and
rejected every tested broker bypass. External IPv6 and missing pip controls remain
explicitly inconclusive. Focused regressions passed 141 tests (2 Docker tests
skipped), plus 8 checker/source-filter tests; mypy and pylint passed.
