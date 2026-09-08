#!/bin/bash
# Strict blackbox entrypoint: init firewall, then permanently drop privileges.
#
# There is no uv sync: the image holds no project code, only the standard
# library, numpy, scipy, and the agent CLIs.
set -euo pipefail
IFS=$'\n\t'

# Skipped under unprivileged Apptainer, which cannot grant CAP_NET_ADMIN;
# ROBOCODE_SKIP_FIREWALL=1 is set by apptainer_sandbox.py.
if [ "${ROBOCODE_SKIP_FIREWALL:-0}" = "1" ]; then
    echo "entrypoint: ROBOCODE_SKIP_FIREWALL=1, skipping firewall init" >&2
else
    if [ "$(id -u)" -ne 0 ]; then
        echo "entrypoint: firewall initialization requires root" >&2
        exit 1
    fi
    /usr/local/bin/init-firewall.sh
fi

unset ROBOCODE_FIREWALL_EXTRA_DOMAINS ROBOCODE_FIREWALL_HOST_PORT \
    ROBOCODE_SKIP_FIREWALL

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

# Unprivileged Apptainer runs preserve the host UID.
exec "$@"
