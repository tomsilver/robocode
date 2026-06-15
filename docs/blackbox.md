# Blackbox environment access

In **normal mode**, the sandboxed Claude agent reads the gymnasium environment's
source code, reasons about its dynamics, and writes `approach.py`. **Blackbox
mode** withholds that source entirely. The agent must discover the dynamics,
reward structure, and termination conditions *empirically* by driving a live
environment instance that runs on the **host**, reachable only through a narrow
JSON-over-TCP protocol.

The container never sees environment source, oracles, or solution primitives,
and the protocol is JSON-only (no pickle), so a compromised agent cannot
deserialize arbitrary host objects.

Blackbox mode is enabled per approach with the `blackbox: true` config flag
(default `false`) and requires `env_cfg` to be set. It is supported by both
`AgenticApproach` and `AgenticCDLApproach`.

## The two processes

The work is split across a host-side server and a sandbox-side client.

### Host env-server

Deliberately split into two modules so the heavy environment imports stay out
of the main experiment process:

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
   apptainer and local backends.
4. Mounts a filtered copy of the repo that strips `environments/`, the kinder
   `envs/` and `demos/`, plus the always-excluded `oracles/`, `primitives/`,
   `tests/`, and `docs/`.

The Docker firewall stays default-deny but already includes an allow rule for
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

The blackbox wiring (env server, env_client, env_spaces.json, the proxies) is
backend-agnostic and works for all three backends, but the *isolation* that
makes blackbox meaningful differs:

- **docker** (default): full isolation. Env source is stripped from the filtered
  mounts, and the container reaches the host env server via
  `host.docker.internal` behind the default-deny firewall (plus the host `/24`
  allow rule above). The `blackbox_proxy_module_escape` and `blackbox_*`
  red-team tests cover this.
- **apptainer**: real isolation too. Same stripped filtered mounts, and
  `apptainer exec` is run with `--no-home` (the host home, hence the real repo
  under it, is never mounted), `--cleanenv`, and `--pwd /sandbox`, with only the
  filtered `src`/`kindergarden` + sandbox bound. It reaches the env server over
  `127.0.0.1` (apptainer shares the host network namespace, so no
  `--add-host`/firewall is needed; note this also means apptainer does not apply
  the default-deny network firewall). Run
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
  There is no way to make an OS sandbox withhold host reads, so this is left as a
  documented best-effort mode rather than a hard failure; the approach `__init__`
  logs a warning when `blackbox` is combined with the `local` backend. Use it
  only for quick local iteration, not for results that depend on the agent not
  having read the source. Use `docker` or `apptainer` for enforced isolation.

## MCP render tools in blackbox

The visual-debug tools (`render_state`, `render_policy`, under the MCP server
named `robocode-tools`) have two implementations, selected at MCP-config time by
`setup_mcp_config(..., blackbox=...)`:

- **Normal:** `python -m robocode.mcp.local_render --env-config ...` renders
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

## Diagram

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
                      |  127.0.0.1:port (apptainer/local)      | (rw)
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

## Key takeaways

- **One protocol, two consumers inside the sandbox.** The agent's own test
  scripts (via `env_client`) and the blackbox MCP render server (via
  `BlackboxEnv`) both speak the same JSON-lines TCP protocol to the single host
  runtime.
- **Renders cross the boundary as files, not bytes.** PNGs are written by the
  host into the shared `mcp_renders/` mount; only relative paths travel back over
  the socket.
- **Isolation is layered.** Withheld source mounts, JSON-only codec, per-run
  token, per-connection fresh env, and a default-deny firewall. The `local`
  backend is best-effort only.

## Threat model and limits

Blackbox mode is a *methodological* constraint first: it stops the agent from
reading environment source so it must discover the dynamics empirically. The
isolation behind it (withheld mounts, JSON-only codec, per-run token,
per-connection env, default-deny firewall) is real. The host never executes
agent code: the only things it runs are env stepping and `render_state`, both
trusted. `render_policy` deliberately runs in the container, so an agent that
writes a malicious `approach.py` cannot reach the env source through rendering
(there is no env source in the container, and the host never execs the file).
The `blackbox_render_*` red-team tests exercise exactly this path.

One limit is worth stating plainly:

- **The env server listens on all interfaces.** It binds
  `0.0.0.0:<ephemeral>` and the container firewall opens the host's `/24`, so
  the port is reachable from that LAN segment, not just the container. The
  per-run 32-hex token is the only access control on the port.
