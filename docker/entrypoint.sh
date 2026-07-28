#!/bin/bash
# Container entrypoint: uv sync → init firewall → permanently drop privileges.
#
# The firewall requires NET_ADMIN / NET_RAW capabilities:
#   docker run --cap-add=NET_ADMIN --cap-add=NET_RAW ...
#
# src/ and third-party/kindergarden/ are bind-mounted at runtime, so uv sync
# runs here to create the .venv with editable installs pointing at the source.
set -euo pipefail
IFS=$'\n\t'

run_as_node=(
    /usr/bin/setpriv
    --reuid=node
    --regid=node
    --init-groups
    --
)
uv_extra_args=()
if [ -n "${ROBOCODE_UV_EXTRA_ARGS:-}" ]; then
    IFS=' ' read -r -a uv_extra_args <<< "$ROBOCODE_UV_EXTRA_ARGS"
fi

# Install Python deps BEFORE the firewall locks down network access.
# ROBOCODE_UV_EXTRA_ARGS is set (e.g. "--extra bilevel") only when a run needs an
# optional dependency group in the sandbox; unset otherwise so those deps (and
# their source) stay absent.
cd /robocode
if [ "$(id -u)" -eq 0 ]; then
    HOME=/home/node USER=node LOGNAME=node \
        "${run_as_node[@]}" uv sync --frozen --python python3.11 "${uv_extra_args[@]}"
else
    # Unprivileged Apptainer installations may not present fakeroot as UID 0.
    uv sync --frozen --python python3.11 "${uv_extra_args[@]}"
fi
cd /sandbox

# Skipped under apptainer (--fakeroot doesn't grant real CAP_NET_ADMIN for iptables),
# in which case ROBOCODE_SKIP_FIREWALL=1 is set by apptainer_sandbox.py.
if [ "${ROBOCODE_SKIP_FIREWALL:-0}" = "1" ]; then
    echo "entrypoint: ROBOCODE_SKIP_FIREWALL=1, skipping firewall init" >&2
else
    if [ "$(id -u)" -ne 0 ]; then
        echo "entrypoint: firewall initialization requires root" >&2
        exit 1
    fi
    /usr/local/bin/init-firewall.sh
fi

unset ROBOCODE_FIREWALL_EXTRA_DOMAINS ROBOCODE_SKIP_FIREWALL

if [ "$(id -u)" -eq 0 ]; then
    export HOME=/home/node USER=node LOGNAME=node
    exec /usr/bin/setpriv \
        --reuid=node \
        --regid=node \
        --init-groups \
        --bounding-set=-all \
        --inh-caps=-all \
        --ambient-caps=-all \
        --no-new-privs \
        -- "$@"
fi

exec "$@"
