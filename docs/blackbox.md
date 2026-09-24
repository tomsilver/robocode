# Black-box environment access

The paper's main setting is **strict black box**: a coding agent receives a task
description and simulator interaction, with no environment source or injected
primitives. Use all three settings explicitly:
`approach.blackbox=true approach.blackbox_strict=true primitive_level=none`.
The software defaults have not changed: `approach=agentic` alone still gives source
access. See the [README](../README.md) for installation and the paper protocol.

Strict mode is supported by `AgenticApproach` and `AgenticPerInstanceApproach`
under Docker or Apptainer. It provides these boundaries:

- Generated programs use Python's standard library, NumPy, and SciPy. No
  environment, KinDER, simulator, robotics, geometry, or planning packages are
  mounted. Rendering uses a standalone stdlib MCP server and the same numerical
  interpreter; there is no separate dependency-rich render environment.
- The host environment server accepts `reset`, `step`, `render_state`, and
  connection closure. Raw state snapshots, state setting, devectorization,
  collision checks, and primitive proxies are rejected. `render_policy` runs the
  policy inside the container and sends visited observations to the host renderer.
- Before scoring, imports reachable from frozen `approach.py` and its siblings
  are checked against the standard library, NumPy, SciPy, and sibling files
  ([strict_blackbox.py](../src/robocode/utils/strict_blackbox.py)). Dynamic import
  mechanisms such as `importlib`, `runpy`, `__import__`, and `sys.modules` are
  rejected. Final scoring runs separately on the host; this check is a
  methodological guardrail, not a hostile-code sandbox.

**Source access** (`blackbox=false`, `blackbox_strict=false`) is an additional
paper condition that allows reading the filtered implementation and reusing its
helpers. **Legacy black box** (`blackbox=true`, `blackbox_strict=false`) withholds
environment source but retains installed dependencies and configured helpers.
See the [preliminary guide](preliminary-experiments.md#non-strict-black-box) for a
comparison with strict mode. `AgenticCDLApproach`
supports this legacy mode but not strict mode.

Build the strict image before a strict run (Docker), or its SIF for the Apptainer
backend:

```bash
bash docker/build_strict_blackbox.sh
bash docker/build_strict_blackbox_sif.sh  # robocode-strict-blackbox.sif
```

Then run, for example:

```bash
python experiments/run_experiment.py \
  approach=agentic approach/backend=claude_opus5 \
  environment=motion2d_generalized primitive_level=none \
  approach.blackbox=true approach.blackbox_strict=true \
  approach.container_backend=docker approach.max_budget_usd=20.0 \
  replicate_seed=42 eval_seed="$EVAL_SEED" num_eval_tasks=100 eval_timeout=60
```

Set `EVAL_SEED` as in the README before running the example. The rest of this
document preserves the broader legacy protocol for developers; strict-mode
restrictions take precedence over its helper APIs.

## The two processes

The work is split across a host-side server and a sandbox-side client.

### Host env-server

Deliberately split into two modules to avoid an import cycle: the approaches
import `env_server`, and the runtime's render/primitive imports reach back into
the approaches, so folding them together would loop. Keeping them separate also
keeps those imports out of the runtime's import-clean API layer:

- `utils/env_server.py`: lightweight, import-clean API layer. Owns the codec
  (`encode` / `decode`, with numpy arrays tagged as
  `{"__ndarray__": [...], "dtype": ...}`), `serialize_space()`, and the
  `env_server_running()` context manager. Imports no environment code.
- `utils/env_server_runtime.py`: the actual subprocess, launched as
  `python -m robocode.utils.env_server_runtime`. A `ThreadingTCPServer` bound to
  `0.0.0.0:0` (OS-assigned ephemeral port). Heavy imports (env, matplotlib,
  imageio, render primitives) live only here.

The runtime gives **each TCP connection its own fresh env instance**, so the
agent's parallel test scripts do not collide. Every request is checked against a
per-run 32-hex token (`secrets.token_hex(16)`). Errors return
`{"error": "Type: msg"}` with the full traceback logged host-side only, so no
source frames leak to the agent.

### Sandbox client

`utils/env_client.py` is copied into the sandbox as an init file. Its
`make_env()` reads `env_spaces.json`, opens a TCP socket, and returns a gym-like
`BlackboxEnv` exposing `reset`, `step`, `get_state`, `set_state`,
`check_action_collision`, `make_primitives`, `render_state`, `render_policy`,
and `close`. Test scripts import it. **`approach.py` must not import it**,
since the generated approach has to run later without the server.

In strict mode, interaction is limited to `reset`, `step`, `close`, generic space
metadata, and state rendering. The server independently rejects `get_state`,
`set_state`, devectorization/vectorization, collision checking, and remote
primitive calls. `render_state` is allowed; `render_policy` is a container-side
rollout followed by rendering, not a remote policy-execution command.

`make_primitives()` rebuilds the same `primitives` dict the eval harness passes
to `GeneratedApproach`: env-dependent primitives (currently just
`check_action_collision`) proxy to the host over the wire, while generic ones
(e.g. `csp`, `BiRRT`) are imported from their copies under `primitives/`. The
host-side specs come from `robocode.primitives.blackbox_primitive_manifest`,
written into `env_spaces.json`.

## Wire protocol

JSON-lines over TCP: one JSON object per line, terminated with `\n`, each request
carrying the auth token.

| Command | Request fields | Response |
|---|---|---|
| `reset` | `seed`, `options` | `{obs, info}` |
| `step` | `action` | `{obs, reward, terminated, truncated, info}` |
| `get_state` | (none) | `{state}` |
| `set_state` | `state` | `{ok: true}` |
| `check_action_collision` | `state`, `action` | `{collision}` |
| `render_state` | `seed`, `state`, `label` | `{path}` (relative) |
| `devectorize` | `obs` | `{result}` (a handle to an `ObjectCentricState`) |
| `vectorize` | `state` (a handle) | `{result}` (a flat ndarray) |
| `getattr` | `target`, `name` | `{result}` (handle-encoded) |
| `call` | `target`, `args`, `kwargs` | `{result}` (handle-encoded) |
| `close` | (none) | connection closes |

`check_action_collision` runs the env-dependent collision primitive on the host
(it needs the env source) against this connection's env, saving and restoring
state so it has no side effects; the sandbox reaches it via
`make_primitives()`. It is the same surface as `step` (it steps the env with an
agent-supplied action), so it adds no new code-execution path.

There is no `render_policy` command: the client runs the policy episode itself
and renders each visited state with `render_state` (see below), so the server
never executes agent code.

Numpy arrays are encoded as `{"__ndarray__": [...], "dtype": "..."}`. Errors come
back as `{"error": "ExceptionType: message"}`.

### Remote object handles

`devectorize`, `vectorize`, `getattr`, and `call` form a small remote-object
proxy so host-side Python objects can be used from the sandbox *by reference*.
The agent never receives a host object's bytes: anything that is not a JSON
scalar, ndarray, list, dict, or set is stored in a **per-connection handle
registry** on the host and represented on the wire by an opaque
`{"__handle__": "h0", "type": "..."}` token (the `type` field is informational).
A later request names the handle and the host resolves it from the registry.
Client-side, a handle becomes a `_RemoteHandle` whose attribute access and calls
issue `getattr` / `call` requests, so `ocs.get_object_from_name("robot")` and
`crv.plan_crv_actions(ocs, cfg, ...)` work transparently. Sets (e.g.
`get_object_names()`) are tagged `{"__set__": [...]}` so the agent sees a real
`set`, identical to eval. The registry is dropped when the connection closes.

This is what makes `observation_space.devectorize(obs)` / `vectorize(ocs)` and
the `crv_motion_planning` / `crv_motion_planning_grasp` primitives usable in
black-box mode: the CRV module source imports the withheld env, so it is *not*
copied into the sandbox; instead the sandbox calls into it on the host through a
remote-module proxy. Because the real `observation_space` (a relational-structs
`ObjectCentricBoxSpace`) and the real CRV modules are passed to
`GeneratedApproach` at eval, **`approach.py` is identical** whether run in the
sandbox (proxied) or at eval (native).

**Security.** This path runs on the host, which has the full filesystem and the
env source, so the reachable surface is an explicit **allowlist**, not a
denylist. A denylist such as "block dunder attributes" is insufficient: the
planner modules do `import numpy as np` and import `kinder.envs` types at module
scope, so a non-underscore attribute like `crv_motion_planning.np` would hand the
agent the live numpy module (`np.load` with pickle is host RCE; `np.fromfile` /
`save` are host file I/O), and `crv_motion_planning.SE2Pose` would leak the very
env types black-box withholds. The guards are: (1) a module target must be a
**whitelisted** short name (`crv_motion_planning`, `crv_motion_planning_grasp`),
and the attribute must be in that module's **public-API allowlist** (the planner
entry points and `CRVConfig` / `CRVActionLimits` / `RelativeGraspPose`); (2) a
handle target's attribute must be in a **per-type allowlist** keyed by the domain
type (`ObjectCentricState`, `Object`, `Type`, and the CRV value dataclasses),
each exposing only safe, value-returning members; (3) a `call` target must be a
handle already vetted by a prior allowlisted `getattr`; (4) the registry is
per-connection and cleared on close. Anything outside the allowlists (including
every dunder, `np`, and the imported env types) is refused, bounding reachable
host code to the safe public domain API.

## Wiring per backend

The approach, in `train()`:

1. Validates that `blackbox=True` requires `env_cfg`, and that both the
   observation and action spaces pass `serialize_space()` (fails fast on
   non-serializable spaces).
2. Enters `env_server_running(env_cfg, sandbox_dir)`, which spawns the runtime
   subprocess, waits for it to write its chosen port to a port file, then yields
   `(port, token)`.
3. Writes `sandbox_dir/env_spaces.json`:
   `{host, port, token, observation_space, action_space, max_steps}`.
   `host` is `host.docker.internal` for Docker (mapped via
   `--add-host host.docker.internal:host-gateway`) or `127.0.0.1` for the
   Apptainer and local backends. Apptainer then rewrites the port to a private
   loopback relay pinned to the separately configured host `env_server_port`.
4. In legacy black box, mounts a filtered copy of the repo that strips `environments/`, the kinder
   `envs/` and `demos/`, plus the always-excluded `oracles/`, `primitives/`,
   `tests/`, and `docs/`.

The regular (non-strict) Docker firewall stays default-deny but already includes an allow rule for
the host's `/24` (derived from the default gateway, in
`docker/init-firewall.sh`, not added by blackbox mode) so the container can
reach the ephemeral env-server port:

```bash
HOST_IP=$(ip route | grep default | awk '{print $3}' | head -1)
HOST_NETWORK=$(echo "$HOST_IP" | sed 's/\.[0-9]*$/.0\/24/')
iptables -A INPUT  -s "$HOST_NETWORK" -j ACCEPT
iptables -A OUTPUT -d "$HOST_NETWORK" -j ACCEPT
```

### Isolation per backend

The environment protocol has Docker, Apptainer, and legacy local wiring.
The experiment runner rejects local generated-code runs; isolation differs by
transport:

- **docker** (default): container filesystem and firewall isolation. Env source is stripped from the filtered
  mounts, and the container reaches the host env server via
  `host.docker.internal` behind the default-deny firewall (plus the host `/24`
  allow rule above). The `blackbox_proxy_module_escape` and `blackbox_*`
  red-team tests cover this.
- **apptainer**: filesystem and disconnected-network isolation. In legacy mode
  it uses filtered source mounts; strict mode omits project source entirely.
  `apptainer exec` uses `--containall`, `--cleanenv`, `--no-home`, and
  `--pwd /sandbox`, with explicit sandbox, prepared-dependency (regular mode),
  session, and broker/relay mounts. `--no-home` alone is *not* enough to withhold the env source:
  many `apptainer.conf` setups still bind the host `/home`, so the agent could
  read the real source straight off `/home/<user>/.../environments`.
  `--containall` is what fixes this, dropping all default binds (home, tmp, cwd)
  so the stripped source is the only source present. Its network namespace has
  only loopback. A Unix-socket relay reaches one pinned host env-server port, and
  a separate broker permits validated model inference. See
  [network isolation](apptainer-network-isolation.md). Run
  `python integration_tests/red_team_sandbox.py --apptainer-blackbox` (needs
  `robocode-sandbox.sif` built) to verify env source stays unreachable.
- **local**: best-effort ONLY, isolation is NOT enforced. The OS-level sandbox
  (`utils/sandbox.py`) restricts filesystem *writes* to the sandbox dir but
  allows *reads* of the whole host filesystem, so a `local` agent can read the
  env source directly (`cat src/robocode/environments/...`,
  `import robocode.environments`) and bypass the blackbox premise entirely. It
  also does not copy a `primitives/` dir into the sandbox, so
  `env_client.make_primitives()` cannot import the generic primitives there
  (the host-proxy and remote-module primitives still work over the wire, and the
  agent can use `robocode.primitives` directly since the repo is readable).
  This retained low-level implementation is not a supported experiment-runner
  path: `validate_eval_seed_isolation()` rejects `container_backend=local` for
  generated-code methods because experimenter-only files remain readable. Use
  Docker or Apptainer for these experiments.

Strict mode runs under Docker or Apptainer during synthesis. Under Docker its
firewall allows the model provider plus only the exact host TCP port of the
environment server; GitHub, SSH, package registries, and other host ports are not
allowed. Under Apptainer the dependency-clean `robocode-strict-blackbox.sif` runs
in a disconnected namespace with the sandbox, session directory, and read-only
broker socket directory mounted. Only validated model inference and the pinned
environment-server relay cross that network boundary.

Final scoring currently runs on the host. Its import allowlist check is a
methodological guardrail, not a network or hostile-code sandbox. The check
runs before the program is loaded, so an approach that imports `pybullet_helpers`,
`tomsgeoms2d`, `robocode`, `kinder`, or any other undeclared dependency fails the
run with a message naming the import instead of silently succeeding from the host
environment.

## MCP render tools in blackbox

The visual-debug tools (`render_state`, `render_policy`, under the MCP server
named `robocode-tools`) have two implementations, selected at MCP-config time by
`setup_mcp_config(..., blackbox=...)`:

- **Source access:** `python -m robocode.mcp.local_render --env-config ...` renders
  in-process, which needs the env source. Its render code lives in
  `robocode/rendering/` (not the `primitives/` package), so it can render
  without importing the sandbox-stripped `robocode.primitives`; the source-free
  metadata it still needs (e.g. `PRIMITIVE_NAME_TO_FILE`) comes from
  `robocode.primitive_specs`. `render_policy` builds its primitives dict from
  the in-sandbox copied top-level `primitives/` package (the subset the sandbox
  setup copies), not from `robocode.primitives`.
- **Blackbox:** `python -m robocode.mcp.server --env-spaces ...`. This server
  cannot import env source, so its tool implementations hold a `BlackboxEnv`.
  `render_state` **proxies to the host env-server** over the JSON-over-TCP
  protocol; the runtime renders the PNG into the bind-mounted
  `sandbox_dir/mcp_renders/`, returns a *relative* path, and the MCP server
  rewrites it to an absolute `/sandbox/...` path for the agent. `render_policy`
  runs the episode **in the container** (it execs the sandbox's `approach.py`
  and steps the env over the protocol), then renders each visited state via
  `render_state`. The host therefore never executes `approach.py`.

Strict blackbox uses that same proxy protocol, with a standalone stdlib MCP
HTTP server at `/opt/robocode-render/strict_server.py`. Both rendering and agent
scripts use `/opt/robocode-strict/bin/python`; no project or MCP framework package
is installed. A second virtualenv would not prevent agents from importing its
packages by changing `sys.path`. Its host connection permits `render_state` but still rejects raw
`get_state` snapshots and all other helpers. Consequently, strict `render_policy`
renders the observations returned by `reset`/`step` rather than requesting hidden
state snapshots.

## Legacy protocol diagram

This diagram shows the helper-rich protocol and regular Docker firewall. Strict
mode removes helper commands and project mounts and pins the host connection to
the environment-server port. Apptainer uses the broker/relay described above.

```
                              HOST
  +---------------------------------------------------------------------+
  |  experiment process  (AgenticApproach.train)                        |
  |    * validates spaces, mints 32-hex token                           |
  |    * env_server_running(env_cfg, sandbox_dir)  --spawns--+          |
  |    * writes  sandbox_dir/env_spaces.json {host,port,token,spaces}   |
  |                                                          |          |
  |   utils/env_server.py  (codec / lifecycle, NO env imports)          |
  |                                                          v          |
  |   +--------------------------------------------------------------+  |
  |   | env_server_runtime  subprocess                               |  |
  |   |   python -m robocode.utils.env_server_runtime                |  |
  |   |   ThreadingTCPServer @ 0.0.0.0:<ephemeral>                   |  |
  |   |   * token-checked   * fresh env PER connection               |  |
  |   |   * reset/step/get_state/set_state                           |  |
  |   |   * render_state (trusted; no agent code) -> writes PNG --+  |  |
  |   |   * imports env + matplotlib + render prims     |            |  |
  |   +---------------^---------------------------------+------------+  |
  |                   | JSON-lines over TCP             | writes        |
  |                   | (token auth, ndarray-tagged)    v               |
  |                   |                       sandbox_dir/mcp_renders/*.png
  +-------------------+---------------------------------------^---------+
                      |  host.docker.internal:port (docker)   | bind mount
                      |  127.0.0.1:port (local only)          | (rw)
  ====================+======= container boundary (firewall:   | =========
                      |         default-DROP + allow host /24) |
                      |                                        |
  +-------------------+----------------------------------------+--------+
  |  SANDBOX (/sandbox, bind-mounted; env SOURCE withheld)     |        |
  |                   |                                        |        |
  |   Claude agent ---+---- writes --> approach.py  (must NOT import    |
  |      |            |                              env_client)        |
  |      | spawns test scripts                                          |
  |      |   from env_client import make_env                            |
  |      |   env = make_env()  --reads--> env_spaces.json               |
  |      |            +----------------------------> (TCP to host)      |
  |      |                                                              |
  |      | MCP stdio                                                    |
  |      v                                                              |
  |   robocode.mcp.server (blackbox)        reads env_spaces.json       |
  |     render_state -> BlackboxEnv ----------> (same TCP to host) -----+
  |     render_policy -> execs approach.py here, steps env over TCP,    |
  |                      renders each visited state via render_state    |
  |       +- rewrites "mcp_renders/x.png" -> "/sandbox/mcp_renders/x.png"
  +---------------------------------------------------------------------+

  Mounts into sandbox (blackbox-specific):  env_client.py , env_spaces.json
  Withheld:  environments/ , kinder envs+demos , oracles/ , primitives/ , tests/ , docs/
```

Apptainer uses the same JSON protocol through a different transport: private
container loopback → mounted Unix socket → one pinned host environment-server
port. Its namespace has no external interface. The diagram's Docker firewall
and direct local connection do not describe Apptainer's network boundary.

## Key takeaways

- **One protocol, two consumers inside the sandbox.** The agent's own test
  scripts (via `env_client`) and the blackbox MCP render server (via
  `BlackboxEnv`) both speak the same JSON-lines TCP protocol to the single host
  runtime.
- **Renders cross the boundary as files, not bytes.** PNGs are written by the
  host into the shared `mcp_renders/` mount; only relative paths travel back over
  the socket.
- **Isolation is layered.** Withheld source mounts, JSON-only codec, per-run
  token, per-connection fresh env, and Docker's firewall or Apptainer's
  disconnected namespace with restricted relays. The `local`
  backend remains available at the low level but is rejected by the experiment runner.

## Red-teaming strict blackbox

Build the strict image, then run its dedicated live adversarial suite:

```bash
bash docker/build_strict_blackbox.sh
python integration_tests/red_team_sandbox.py --strict-blackbox
# Apptainer: the same suite, including network and package-install probes.
bash docker/build_strict_blackbox_sif.sh
python integration_tests/red_team_sandbox.py --apptainer-strict-blackbox
```

The suite first proves the allowed surface works: the generated-program
interpreter can import NumPy and SciPy and can reset and step the environment, and
the same dependency-clean interpreter renders a state through the host.
It then asks an agent to attack the same configuration used in production and
fails if any of these boundaries break:

- the generated-program interpreter imports RoboCode, KinDER, MCP, Gymnasium,
  geometry, or simulator packages, or installs a package from the network;
- changing interpreters/package paths, or executing a policy through MCP,
  exposes withheld project, framework, environment, or simulator packages;
- environment source, the host-side canary, or arbitrary Internet content is
  reachable through filesystem or network probing;
- withheld client methods or raw protocol commands such as `get_state`,
  `set_state`, devectorization, remote attribute access, or collision helpers
  succeed against the strict env server.

This is a live agent test and consumes model budget. The deterministic companion
coverage in `tests/utils/test_strict_blackbox.py` checks the strict server surface,
dependency-clean image wiring, import allowlist (including transitive and dynamic
import attempts), cached-module shadowing, and per-sandbox sibling-module isolation.
The live suite attacks the synthesis boundary; the deterministic tests exercise the
host-side scoring gate. That gate is a cooperative methodological guardrail, not a
hostile-code sandbox, and does not claim to contain deliberately obfuscated Python
after scoring begins.

## Threat model and limits

Blackbox mode is a *methodological* constraint first: it stops the agent from
reading environment source so it must discover the dynamics empirically. The
isolation behind it (withheld mounts, JSON-only codec, per-run token,
per-connection env, and backend-specific network restrictions) applies during
agent execution. The environment server runs trusted env stepping and
`render_state`, never agent code. `render_policy` deliberately runs in the container, so an agent that
writes a malicious `approach.py` cannot reach the env source through rendering
(there is no env source in the container, and the environment server never
executes the file).
The `blackbox_render_*` red-team tests exercise exactly this path.

Final policy scoring is a separate host-side execution path and is not contained
by these agent-runtime protections. Full experiment network isolation requires
addressing that path too.

Another limit:

- **The env server listens on all host interfaces** at `0.0.0.0:<ephemeral>`.
  Its reachability from other machines depends on the host firewall; requests
  require the per-run token. The regular Docker container permits host `/24`
  connections, strict Docker pins the environment-server port, and Apptainer
  uses a pinned relay. Container egress rules do not restrict unrelated LAN
  clients connecting to a host listener.
