# Apptainer agent isolation

The Apptainer backend runs Codex and Claude in a disconnected network namespace.
It requires an installation that permits unprivileged user/network namespaces and
`--network none`. The launcher checks isolation before starting the agent and
aborts if the required boundary cannot be established.

## Setup

Build the image used by your experiment:

```sh
bash docker/build_sif.sh
# For strict blackbox experiments:
bash docker/build_strict_blackbox_sif.sh
```

Configure `approach.container_backend=apptainer` and authenticate the selected
backend on the host. Credentials must be valid when a run starts; refresh/login
happens on the host. OpenCode, custom model endpoints, and the separate
GenPlan/Best-of-K Apptainer path are unsupported and fail closed.

Use an updated launcher and rebuild strict images when their Dockerfile or copied
render sources change. Updating only the image does not update launcher code in
another checkout. The launcher rejects the legacy strict MCP environment.

## Permitted connections

The agent has only loopback networking. Two mounted Unix sockets provide the
connections needed for experiments:

- The **model broker** holds real credentials on the host and forwards validated
  inference requests to fixed provider endpoints. It rejects arbitrary destinations,
  CONNECT, redirects, hosted web/MCP tools, remote content references, and unsupported
  operations. Client-side tool definitions and inline raster images are supported.
  Logs contain endpoint/status/rejection information, not request bodies or tokens.
- The **environment relay** reaches one host port selected by trusted launcher
  configuration (`env_server_port`). It supports the existing reset/step/render
  protocol. Agent-writable `env_spaces.json` cannot select a new host destination.

Inference is an intentional external connection. Agents can submit permitted
inference requests themselves; the broker is not a spending or rate-limit boundary.

## Runtime boundaries

The supervisor requires a non-root UID, loopback-only interfaces, no IPv4 routes,
zero capabilities, and no-new-privileges. Filesystem/PID isolation and a clean
process environment prevent default host-home mounts and inherited credentials.
There is no fallback to host networking.

Regular dependencies are installed in a trusted preparation phase with network
access, before agent execution and without agent files, sessions, or credentials.
The completed environment is mounted read-only. Runtime offline environment
variables supplement the network namespace; they do not enforce isolation alone.

Strict images contain stdlib, NumPy/SciPy, and generic protocol helpers, without
project or MCP framework packages. Both MCP render tools use the strict interpreter;
agent-written policies execute inside the container. A separate virtualenv would
not stop agents from importing its readable packages via another `sys.path`.
Filtered source mounts also exclude compiled bytecode for withheld modules.

The trusted host, kernel, Apptainer, broker, dependency preparation, and environment
server remain part of the boundary. Final scoring runs separately on the host;
its import allowlist is a methodological guardrail, not hostile-code containment.

## Code and Docker behavior

| Module | Responsibility |
| --- | --- |
| `apptainer_sandbox.py` | Agent launch, CLI configuration, broker/relay lifetime |
| `model_broker.py` | Credentials, fixed provider endpoints, request validation |
| `isolated_transport.py` | Namespace checks and fixed-destination relays |
| `apptainer_environment.py` | Dependency preparation and clean child environment |

Provider hostname constants are shared with Docker's registry. Docker retains its
firewall and credential transport; `ROBOCODE_FIREWALL_EXTRA_DOMAINS` configures Docker,
not the Apptainer broker. Strict Docker images use the same dependency-clean render
server and must also be rebuilt after render-image changes.

Audit the actual node, runtime, and images before relying on isolation, and repeat
after relevant changes. See [red-teaming instructions](apptainer-red-teaming.md)
for controlled network checks and agent probes. Missing tools, unavailable positive
controls, provider refusals, and incomplete reports are not successful isolation tests.
