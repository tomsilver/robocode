"""Check that the env server survives concurrent connections against a 3D env.

Usage:
    python integration_tests/check_env_server_concurrency.py
    python integration_tests/check_env_server_concurrency.py --connections 8 --waves 8

The env server builds a fresh env per connection on a ``ThreadingTCPServer`` worker
thread, and a MuJoCo env owns a render context that is released only by ``__del__``
-- ``env.close()`` does not reach it. Left to the GC, that context is freed on
whatever thread happens to trigger the next collection, and freeing a CGL context
off its owning thread segfaults on macOS:

    glDeleteTextures <- mjr_freeContext <- slot_tp_finalize <- gc_collect_main

That killed the server mid-run twice, on balancebeam3d and sweepintodrawer3d. Both
times the agent kept working against the dead server and the whole run was discarded
after the budget had been spent.

The agent triggers this by doing what it is told it may do -- run test scripts in
parallel, e.g. a ThreadPoolExecutor calling make_env() eight times. This opens that
many connections at once and closes them together, which is what the waves below
imitate. It is a race, so a single wave proves little; the default is deliberately
several.

Requires the kinder dynamic3D extras. Raises on any failure.
"""

import argparse
import json
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any

from robocode.utils.env_server import env_server_running

# A 3D env, since the crash is in the MuJoCo render context. Mirrors
# experiments/conf/environment/balancebeam3d.yaml.
_ENV_CFG: dict[str, Any] = {
    "_target_": "robocode.environments.kinder_geom3d_env.KinderGeom3DEnv",
    "env_id": "kinder/BalanceBeam3D-o3-v0",
    "scene_bg": False,
}


def _exercise_connection(port: int, token: str, seed: int, errors: list[str]) -> None:
    """Open one connection, build an env on the server, use it, and close it."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=300) as sock:
            stream = sock.makefile("rwb")

            def call(**request: Any) -> dict[str, Any]:
                stream.write((json.dumps({"token": token, **request}) + "\n").encode())
                stream.flush()
                line = stream.readline()
                if not line:
                    raise RuntimeError("server closed the connection mid-request")
                return dict(json.loads(line))

            call(cmd="reset", seed=seed)
            call(cmd="get_state")
            # "close" is answered by hanging up, not by a reply, so send it and
            # let the socket close rather than waiting for a response that the
            # handler never writes.
            stream.write((json.dumps({"token": token, "cmd": "close"}) + "\n").encode())
            stream.flush()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        errors.append(f"seed {seed}: {type(exc).__name__}: {exc}")


def main() -> None:
    """Hammer the server with concurrent connections and assert it stays up."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connections", type=int, default=8, help="per wave")
    parser.add_argument("--waves", type=int, default=4)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        sandbox_dir = Path(tmp) / "sandbox"
        sandbox_dir.mkdir()
        errors: list[str] = []
        with env_server_running(json.dumps(_ENV_CFG), sandbox_dir) as (port, token):
            for wave in range(args.waves):
                threads = [
                    threading.Thread(
                        target=_exercise_connection,
                        args=(port, token, wave * 100 + i, errors),
                    )
                    for i in range(args.connections)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                if errors:
                    # The server log is the only place its cause of death is
                    # recorded, and it lives in the temp dir we are about to
                    # delete, so fold it into the failure.
                    log = Path(tmp) / "env_server.log"
                    tail = (
                        "".join(
                            log.read_text(encoding="utf-8").splitlines(keepends=True)[
                                -25:
                            ]
                        )
                        if log.is_file()
                        else "(no env_server.log)"
                    )
                    raise AssertionError(
                        f"wave {wave}: the server stopped answering, which is what a "
                        f"crash looks like from a client:\n  "
                        + "\n  ".join(errors[:5])
                        + f"\n--- env_server.log (tail) ---\n{tail}"
                    )
                print(
                    f"OK  wave {wave}: {args.connections} concurrent connections served"
                )
    # Leaving the context manager re-raises if the server died at any point, so
    # reaching here means it survived every wave and shut down cleanly.
    print(
        f"Env server survived {args.waves * args.connections} concurrent connections."
    )


if __name__ == "__main__":
    main()
