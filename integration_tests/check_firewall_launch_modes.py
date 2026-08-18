"""Check that the sandbox firewall initializes in both container launch modes.

Usage:
    python integration_tests/check_firewall_launch_modes.py
    python integration_tests/check_firewall_launch_modes.py --simulate-unresolvable

``docker_sandbox.py`` passes ``--add-host host.docker.internal:host-gateway`` only
for blackbox or local-model runs, so the firewall script must survive that name
being absent. It did not: ``getent`` exits 2 when the name does not resolve, and
under ``set -euo pipefail`` that aborted the script before the default-deny
policies were applied. Since ``entrypoint.sh`` is ``set -e`` too, the container
died at startup and the run never reached the agent.

This runs the working tree's script -- not the copy baked into the image, so an
edit is checked without a rebuild -- in a fresh container in both modes, and
asserts the two things that must hold regardless: the script completes, and the
resulting rule set denies outbound traffic by default.

Docker Desktop resolves ``host.docker.internal`` whether or not the flag is
passed, so on macOS neither mode exercises the non-resolving path that broke
Linux. ``--simulate-unresolvable`` points the script at a name that resolves
nowhere, which reproduces it on any platform.

Requires a working docker daemon and the ``robocode-sandbox`` image. Raises on any
failure.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

IMAGE = "robocode-sandbox"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIREWALL = _REPO_ROOT / "docker" / "init-firewall.sh"

# The name is mapped for blackbox and local-model runs, and absent otherwise; see
# _docker_run_prefix() in robocode.utils.docker_sandbox.
_LAUNCH_MODES: dict[str, list[str]] = {
    "whitebox (no --add-host)": [],
    "blackbox (--add-host)": ["--add-host", "host.docker.internal:host-gateway"],
}


def _run_firewall(script: Path, extra_args: list[str]) -> tuple[int, str]:
    """Run *script* in a fresh container; return its exit status and rule set."""
    # `;` rather than `&&` so a non-zero status still reports the rule set.
    command = 'bash /fw.sh > /dev/null 2>&1; echo "rc=$?"; iptables -S'
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--cap-add=NET_ADMIN",
            "--cap-add=NET_RAW",
            *extra_args,
            "-v",
            f"{script}:/fw.sh:ro",
            "--entrypoint",
            "bash",
            IMAGE,
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout
    status_lines = [line for line in output.splitlines() if line.startswith("rc=")]
    if not status_lines:
        raise RuntimeError(
            f"firewall script produced no status line.\n"
            f"stdout: {output}\nstderr: {completed.stderr}"
        )
    return int(status_lines[0].removeprefix("rc=")), output


def _check(script: Path) -> None:
    """Assert the firewall completes and denies by default in both launch modes."""
    for mode, extra_args in _LAUNCH_MODES.items():
        status, rules = _run_firewall(script, extra_args)
        if status != 0:
            raise AssertionError(
                f"{mode}: firewall script exited {status}. A non-zero status here "
                f"means the container dies at startup, since entrypoint.sh is "
                f"`set -e`.\n{rules}"
            )
        if "-P OUTPUT DROP" not in rules:
            raise AssertionError(
                f"{mode}: outbound traffic is not denied by default. The script may "
                f"have exited before applying its policies.\n{rules}"
            )
        print(f"OK  {mode}: script completed, -P OUTPUT DROP applied")


def main() -> None:
    """Run the check, optionally against an unresolvable host name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate-unresolvable",
        action="store_true",
        help="Rewrite host.docker.internal to a name that resolves nowhere, "
        "reproducing on macOS the condition that only occurs on Linux",
    )
    args = parser.parse_args()

    if not args.simulate_unresolvable:
        _check(_FIREWALL)
        print("Firewall initializes correctly in both launch modes.")
        return

    source = _FIREWALL.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        patched = Path(tmp) / "init-firewall.sh"
        patched.write_text(
            source.replace("host.docker.internal", "nowhere.invalid"),
            encoding="utf-8",
        )
        _check(patched)
    print("Firewall initializes correctly even when the host name does not resolve.")


if __name__ == "__main__":
    main()
