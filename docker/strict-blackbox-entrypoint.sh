#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

/usr/local/bin/init-firewall.sh
unset ROBOCODE_FIREWALL_EXTRA_DOMAINS ROBOCODE_FIREWALL_HOST_PORT

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
