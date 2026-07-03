#!/bin/bash
# Container entrypoint: init firewall → uv sync → run claude.
#
# The firewall requires NET_ADMIN / NET_RAW capabilities:
#   docker run --cap-add=NET_ADMIN --cap-add=NET_RAW ...
#
# src/ and third-party/kindergarden/ are bind-mounted at runtime, so uv sync
# runs here to create the .venv with editable installs pointing at the source.
#
# The `node` user has passwordless sudo for init-firewall.sh and uv
# (configured in the Dockerfile via /etc/sudoers.d/node-firewall).
set -e

# Install Python deps BEFORE the firewall locks down network access.
# ROBOCODE_UV_EXTRA_ARGS is set (e.g. "--extra bilevel") only when a run needs an
# optional dependency group in the sandbox; unset otherwise so those deps (and
# their source) stay absent.
cd /robocode
# shellcheck disable=SC2086
uv sync --frozen --python python3.11 ${ROBOCODE_UV_EXTRA_ARGS:-}
cd /sandbox

# Pass ROBOCODE_FIREWALL_EXTRA_DOMAINS through sudo (sudo strips env by default).
# Skipped under apptainer (--fakeroot doesn't grant real CAP_NET_ADMIN for iptables),
# in which case ROBOCODE_SKIP_FIREWALL=1 is set by apptainer_sandbox.py.
if [ "${ROBOCODE_SKIP_FIREWALL:-0}" = "1" ]; then
    echo "entrypoint: ROBOCODE_SKIP_FIREWALL=1, skipping firewall init" >&2
else
    sudo ROBOCODE_FIREWALL_EXTRA_DOMAINS="${ROBOCODE_FIREWALL_EXTRA_DOMAINS:-}" \
        /usr/local/bin/init-firewall.sh
fi

exec "$@"
