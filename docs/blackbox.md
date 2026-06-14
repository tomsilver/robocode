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
`render_state`, `render_policy`, and `close`. Test scripts import it.
**`approach.py` must not import it**, since the generated approach has to run
later without the server.

## Wire protocol

JSON-lines over TCP: one JSON object per line, terminated with `\n`, each request
carrying the auth token.

| Command | Request fields | Response |
|---|---|---|
| `reset` | `seed`, `options` | `{obs, info}` |
| `step` | `action` | `{obs, reward, terminated, truncated, info}` |
| `get_state` | (none) | `{state}` |
| `set_state` | `state` | `{ok: true}` |
| `render_state` | `seed`, `state`, `label` | `{path}` (relative) |
| `render_policy` | `seed`, `max_steps`, `max_frames` | `{paths}` (relative) |
| `close` | (none) | connection closes |

Numpy arrays are encoded as `{"__ndarray__": [...], "dtype": "..."}`. Errors come
back as `{"error": "ExceptionType: message"}`.

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

The Docker firewall stays default-deny but adds an allow rule for the host's
`/24` (derived from the default gateway) so the container can reach the
ephemeral env-server port:

```bash
HOST_IP=$(ip route | grep default | awk '{print $3}' | head -1)
HOST_NETWORK=$(echo "$HOST_IP" | sed 's/\.[0-9]*$/.0\/24/')
iptables -A INPUT  -s "$HOST_NETWORK" -j ACCEPT
iptables -A OUTPUT -d "$HOST_NETWORK" -j ACCEPT
```

The `local` backend is supported but logged as best-effort only: an OS sandbox
cannot prevent reading env source straight off the host filesystem.

## MCP render tools in blackbox

The visual-debug tools (`render_state`, `render_policy`, under the MCP server
named `robocode-tools`) have two implementations, selected at MCP-config time by
`setup_mcp_config(..., blackbox=...)`:

- **Normal:** `python -m robocode.mcp.local_render --env-config ...` renders
  in-process, which needs the env source.
- **Blackbox:** `python -m robocode.mcp.server --env-spaces ...`. This server
  cannot import env source, so its tool implementations hold a `BlackboxEnv` and
  **proxy the render call to the host env-server over the same JSON-over-TCP
  protocol**. The runtime renders the PNG into the bind-mounted
  `sandbox_dir/mcp_renders/`, returns a *relative* path, and the MCP server
  rewrites it to an absolute `/sandbox/...` path for the agent.

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
  |   |   * render_state/render_policy -> writes PNGs --+            |  |
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
  |     render_state / render_policy                                    |
  |       +- BlackboxEnv ----------------------> (same TCP to host) ----+
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
