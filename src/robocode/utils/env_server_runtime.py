"""Black-box environment server runtime (the subprocess entry point).

Launched by :func:`robocode.utils.env_server.env_server_running` as
``python -m robocode.utils.env_server_runtime``. This module holds the
serving loop, the render_state handler, and the env-dependent
check_action_collision primitive, all of which depend on the environment
source (and render needs matplotlib/imageio). Keeping it separate from
``env_server`` (the import-facing API used by the approaches) means the main
experiment process never transitively imports any of that heavy code.

The policy render (``render_policy``) runs entirely in the sandbox, not
here: the client drives the env step by step and asks this server only to
render the states it visits via ``render_state``. That way no agent code
runs host-side, so a black-box agent cannot reach the env source through
rendering.

See :mod:`robocode.utils.env_server` for the wire protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import secrets
import socketserver
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from relational_structs import Object, ObjectCentricState, Type

from robocode.primitives import crv_motion_planning, crv_motion_planning_grasp
from robocode.primitives.check_action_collision import check_action_collision
from robocode.primitives.crv_motion_planning import CRVActionLimits, CRVConfig
from robocode.primitives.crv_motion_planning_grasp import (
    CRVGraspWaypoint,
    RelativeGraspPose,
)
from robocode.rendering.render_state import render_state as render_state_fn
from robocode.utils.env_server import decode, encode, serialize_space

logger = logging.getLogger(__name__)

# Short module names the agent may target with a remote-module proxy. Only these
# modules are reachable through the getattr/call commands; any other name is
# rejected (see _resolve_getattr_target). Mirrors
# robocode.primitives.REMOTE_MODULE_PRIMITIVES.
_REMOTE_MODULES: dict[str, Any] = {
    "crv_motion_planning": crv_motion_planning,
    "crv_motion_planning_grasp": crv_motion_planning_grasp,
}

# SECURITY ALLOWLISTS. The getattr/call commands run on the host, which has the
# full filesystem and the env source, so the reachable surface must be an
# explicit allowlist, NOT a denylist. A denylist (e.g. "block dunders") is
# insufficient: the planner modules do ``import numpy as np`` and import
# kinder.envs types at module scope, so a non-underscore attribute like
# ``crv_motion_planning.np`` would otherwise hand the agent the live numpy
# module (np.load with pickle is host RCE; np.fromfile/save are host file I/O)
# or the very kinder.envs classes blackbox withholds. So we expose only the
# named public planner API per module, and only named methods/attrs per domain
# type. Anything else is refused.
_REMOTE_MODULE_API: dict[str, frozenset[str]] = {
    "crv_motion_planning": frozenset(
        {
            "plan_crv_actions",
            "plan_crv_base_actions",
            "plan_crv_holding_actions",
            "crv_action_plan_to_pose_plan",
            "crv_pose_plan_to_action_plan",
            "CRVConfig",
            "CRVActionLimits",
        }
    ),
    "crv_motion_planning_grasp": frozenset(
        {
            "plan_crv_grasp",
            "RelativeGraspPose",
        }
    ),
}

# Per-type method/attribute allowlist for handles, checked by isinstance. These
# are the only domain objects a handle ever wraps (devectorize results, planner
# inputs/outputs); each exposes only safe, public, value-returning members.
_HANDLE_API: tuple[tuple[type, frozenset[str]], ...] = (
    (
        ObjectCentricState,
        frozenset(
            {
                "get",
                "set",
                "copy",
                "get_object_from_name",
                "get_objects",
                "get_object_names",
                "vec",
            }
        ),
    ),
    (Object, frozenset({"name", "type", "is_instance"})),
    (Type, frozenset({"name"})),
    (CRVConfig, frozenset({"x", "y", "theta"})),
    (CRVActionLimits, frozenset({"max_dx", "max_dy", "max_dtheta"})),
    (RelativeGraspPose, frozenset({"x", "y", "theta"})),
    (CRVGraspWaypoint, frozenset({"x", "y", "theta", "arm_joint", "vacuum"})),
)

_HANDLE_TAG = "__handle__"
_NDARRAY_TAG = "__ndarray__"
_MODULE_TAG = "__module__"
_SET_TAG = "__set__"

# render_state_fn (kinder render_2dstate) drives the global pyplot state machine
# (plt.subplots / plt.tight_layout / plt.close), which is not thread-safe. Each
# connection runs in its own ThreadingTCPServer thread, so serialize the
# matplotlib render across connections. Env stepping stays concurrent.
_RENDER_LOCK = threading.Lock()


class _HandleRegistry:
    """Per-connection table of host objects the agent holds by reference.

    The agent never receives host objects themselves; for anything that is not a
    JSON scalar/ndarray/container, ``encode_ref`` stores the object here and
    sends back an opaque ``{"__handle__": id}`` token. A later request that names
    the handle resolves it through ``get``. The registry lives only for the
    connection and is dropped when it closes, so handles cannot leak across
    connections or outlive the env.
    """

    def __init__(self) -> None:
        self._objects: dict[str, Any] = {}
        self._counter = 0

    def register(self, obj: Any) -> str:
        """Store *obj* and return a fresh handle id."""
        handle_id = f"h{self._counter}"
        self._counter += 1
        self._objects[handle_id] = obj
        return handle_id

    def get(self, handle_id: str) -> Any:
        """Look up a handle, raising loudly if it is unknown."""
        if handle_id not in self._objects:
            raise ValueError(f"Unknown handle id: {handle_id!r}")
        return self._objects[handle_id]

    def clear(self) -> None:
        """Drop all handles (called when the connection closes)."""
        self._objects.clear()


def encode_ref(obj: Any, registry: _HandleRegistry) -> Any:
    """Encode *obj* for the wire, registering non-JSON objects as handles.

    Scalars, strings, booleans, and None go inline; numpy scalars become their
    Python value; numpy arrays use the existing ``{"__ndarray__": ...}`` tag;
    lists/tuples and str-keyed dicts recurse. Anything else (an
    ObjectCentricState, Object, Type, CRVConfig, planner result, ...) is stored
    in *registry* and represented by an opaque handle. The ``type`` field is
    informational only, so the agent can see what kind of object it holds.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {_NDARRAY_TAG: obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, dict) and all(isinstance(key, str) for key in obj):
        return {key: encode_ref(value, registry) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_ref(value, registry) for value in obj]
    if isinstance(obj, (set, frozenset)):
        # Preserve set semantics across the wire (e.g. get_object_names()
        # returns a set); the client rebuilds a set so the agent sees the same
        # type it gets at eval.
        return {_SET_TAG: [encode_ref(value, registry) for value in obj]}
    return {_HANDLE_TAG: registry.register(obj), "type": type(obj).__name__}


def decode_ref(obj: Any, registry: _HandleRegistry) -> Any:
    """Decode a wire value, resolving handles back to their host objects."""
    if isinstance(obj, dict):
        if _HANDLE_TAG in obj:
            return registry.get(obj[_HANDLE_TAG])
        if _NDARRAY_TAG in obj:
            return np.array(obj[_NDARRAY_TAG], dtype=np.dtype(obj["dtype"]))
        if _SET_TAG in obj:
            return {decode_ref(value, registry) for value in obj[_SET_TAG]}
        return {key: decode_ref(value, registry) for key, value in obj.items()}
    if isinstance(obj, list):
        return [decode_ref(value, registry) for value in obj]
    return obj


def _resolve_getattr_target(target: Any, name: str, registry: _HandleRegistry) -> Any:
    """Resolve a getattr target and enforce the allowlist for *name*.

    A target is either a whitelisted module reference (``{"__module__": short_name}``)
    or a registry handle (``{"__handle__": id}``). The attribute *name* is checked
    against the per-module or per-type allowlist BEFORE the attribute is read, so only
    the public planner API and safe domain members are ever reachable. See the allowlist
    comments above for why this must be an allowlist and not a denylist.
    """
    if isinstance(target, dict) and _MODULE_TAG in target:
        short = target[_MODULE_TAG]
        if short not in _REMOTE_MODULES:
            raise ValueError(f"Refusing access to non-whitelisted module: {short!r}")
        if name not in _REMOTE_MODULE_API[short]:
            raise ValueError(
                f"Refusing attribute {name!r} on module {short!r}; only the "
                "public planner API is exposed"
            )
        return _REMOTE_MODULES[short]
    obj = decode_ref(target, registry)
    for cls, allowed in _HANDLE_API:
        if isinstance(obj, cls):
            if name in allowed:
                return obj
            raise ValueError(
                f"Refusing attribute {name!r} on {type(obj).__name__} handle"
            )
    raise ValueError(f"No remote API exposed for {type(obj).__name__} handle")


def _resolve_call_target(target: Any, registry: _HandleRegistry) -> Any:
    """Resolve a call target, which must be a handle vetted by a prior getattr.

    Callable handles are only ever created by an allowlisted ``getattr`` (a
    module function, a domain class like CRVConfig, or a domain method), so a
    handle reaching ``call`` is already vetted. Module references are refused
    here: the agent must fetch a function via ``getattr`` (allowlist-checked)
    first.
    """
    if isinstance(target, dict) and _MODULE_TAG in target:
        raise ValueError("call target must be a handle obtained via getattr")
    return decode_ref(target, registry)


class _EnvServer(socketserver.ThreadingTCPServer):
    """TCP server holding the env config, auth token, and sandbox dir."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self, env_config: dict[str, Any], token: str, sandbox_dir: Path
    ) -> None:
        super().__init__(("0.0.0.0", 0), _EnvRequestHandler)
        self.env_config = env_config
        self.token = token
        self.sandbox_dir = sandbox_dir


class _EnvRequestHandler(socketserver.StreamRequestHandler):
    """Serves one client connection with its own fresh env instance."""

    server: _EnvServer

    def handle(self) -> None:
        logger.info("New connection from %s", self.client_address)
        env = instantiate(OmegaConf.create(self.server.env_config))
        # Reset once so get_state/set_state/render work before the client
        # issues its own reset (mirrors the MCP server's env setup).
        env.reset(seed=0)
        # Per-connection handle table backing the remote-object proxy (used only
        # by the devectorize/vectorize/getattr/call commands).
        registry = _HandleRegistry()
        try:
            for line in self.rfile:
                request = json.loads(line)
                # Constant-time compare; reject anything that is not a matching
                # str (compare_digest requires both operands be str/bytes).
                token = request.pop("token", None)
                if not isinstance(token, str) or not secrets.compare_digest(
                    token, self.server.token
                ):
                    logger.warning("Rejected request with bad token")
                    return
                if request["cmd"] == "close":
                    return
                # Errors from agent-controlled requests are reported back
                # over the wire (without traceback frames, which would leak
                # env source lines) rather than killing the connection. The
                # broad catch is intentional: any failure the agent triggers
                # must reach its test script as a message.
                try:
                    payload = _dispatch(env, request, self.server.sandbox_dir, registry)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "Request %s failed:\n%s", request, traceback.format_exc()
                    )
                    payload = {"error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write(json.dumps(payload).encode("utf-8") + b"\n")
        finally:
            registry.clear()
            env.close()
            logger.info("Connection from %s closed", self.client_address)


def _dispatch(
    env: Any,
    request: dict[str, Any],
    sandbox_dir: Path,
    registry: _HandleRegistry,
) -> dict[str, Any]:
    """Execute one decoded request against the env and encode the reply."""
    cmd = request["cmd"]
    if cmd == "reset":
        obs, info = env.reset(seed=request.get("seed"), options=request.get("options"))
        return {"obs": encode(obs), "info": encode(info)}
    if cmd == "step":
        obs, reward, terminated, truncated, info = env.step(decode(request["action"]))
        return {
            "obs": encode(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": encode(info),
        }
    if cmd == "get_state":
        return {"state": encode(env.get_state())}
    if cmd == "set_state":
        env.set_state(decode(request["state"]))
        return {"ok": True}
    if cmd == "check_action_collision":
        collision = check_action_collision(
            env, decode(request["state"]), decode(request["action"])
        )
        return {"collision": bool(collision)}
    if cmd == "render_state":
        return {
            "path": _render_state(
                env,
                sandbox_dir,
                request.get("seed", 42),
                request.get("state"),
                request.get("label", ""),
            )
        }
    # Remote-object proxy commands. They use the handle-aware encode_ref/
    # decode_ref (NOT the plain encode/decode above) so host-side domain objects
    # can be used from the sandbox by reference.
    if cmd == "devectorize":
        ocs = env.observation_space.devectorize(decode_ref(request["obs"], registry))
        return {"result": encode_ref(ocs, registry)}
    if cmd == "vectorize":
        vec = env.observation_space.vectorize(decode_ref(request["state"], registry))
        return {"result": encode_ref(vec, registry)}
    if cmd == "getattr":
        name = request["name"]
        target = _resolve_getattr_target(request["target"], name, registry)
        return {"result": encode_ref(getattr(target, name), registry)}
    if cmd == "call":
        target = _resolve_call_target(request["target"], registry)
        args = decode_ref(request.get("args", []), registry)
        kwargs = decode_ref(request.get("kwargs", {}), registry)
        return {"result": encode_ref(target(*args, **kwargs), registry)}
    raise ValueError(f"Unknown command: {cmd!r}")


def _safe_label(label: str) -> str:
    """Reduce an agent-supplied label to a filename-safe token.

    The label is interpolated into the render output filename. This handler runs
    on the host (not in the container), so a label with path separators or ``..``
    would let a black-box agent steer the PNG write outside the sandbox. Keep
    only alphanumerics, dashes, and underscores; collapse everything else to
    ``_``.
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label)


def _unique_render_path(directory: Path, stem: str, ext: str = ".png") -> Path:
    """Return ``directory/stem.ext``, appending _1, _2, ...

    if taken.
    """
    candidate = directory / f"{stem}{ext}"
    i = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{i}{ext}"
        i += 1
    return candidate


def _render_state(
    env: Any,
    sandbox_dir: Path,
    seed: int,
    state: list[float] | None,
    label: str,
) -> str:
    """Render a state to a PNG under ``mcp_renders/``; return the relative path.

    Mirrors the in-container render_state MCP tool, but runs on the host where the
    environment source and render code live. The PNG lands in the bind-mounted sandbox
    dir so the container sees it too. Saves and restores the connection's env state so
    the render leaves no side effect (render_state_fn does its own internal
    get_state/set_state/restore from the state we pass it).
    """
    saved = env.get_state()
    try:
        if state is not None:
            env_state = np.array(state, dtype=np.float32)
            stem = "state_custom"
        else:
            env.reset(seed=seed)
            env_state = env.get_state()
            stem = f"state_seed{seed}"
        if label:
            stem += f"_{_safe_label(label)}"

        with _RENDER_LOCK:
            frame = render_state_fn(env, env_state)
        out_dir = sandbox_dir / "mcp_renders"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = _unique_render_path(out_dir, stem)
        iio.imwrite(str(out), frame)
        return str(out.relative_to(sandbox_dir))
    finally:
        env.set_state(saved)


def main() -> None:
    """Parse CLI args and serve forever."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-config", required=True, help="Path to JSON file with Hydra env config"
    )
    parser.add_argument(
        "--token", required=True, help="Auth token required on every request"
    )
    parser.add_argument(
        "--port-file", required=True, help="File to write the chosen port to"
    )
    parser.add_argument(
        "--sandbox-dir",
        required=True,
        help="Sandbox dir where render PNGs and approach.py live",
    )
    args = parser.parse_args()

    env_config = json.loads(Path(args.env_config).read_text(encoding="utf-8"))

    # Fail fast on a bad config or unserializable spaces before serving.
    env = instantiate(OmegaConf.create(env_config))
    serialize_space(env.observation_space)
    serialize_space(env.action_space)
    env.close()

    server = _EnvServer(env_config, args.token, Path(args.sandbox_dir))
    port = server.server_address[1]
    Path(args.port_file).write_text(str(port), encoding="utf-8")
    logger.info("Serving env %s on port %d", env_config.get("_target_"), port)
    server.serve_forever()


if __name__ == "__main__":
    main()
