#!/bin/bash
# Container entrypoint: init firewall → uv sync → run claude.
#
# The firewall requires NET_ADMIN / NET_RAW capabilities:
#   docker run --cap-add=NET_ADMIN --cap-add=NET_RAW ...
#
# src/ and prpl-mono/ are bind-mounted at runtime, so uv sync runs here
# to create the .venv with editable installs pointing at the mounted source.
#
# The `node` user has passwordless sudo for init-firewall.sh and uv
# (configured in the Dockerfile via /etc/sudoers.d/node-firewall).
set -e

# Install Python deps BEFORE the firewall locks down network access.
cd /robocode
uv sync --frozen --python python3.11
cd /sandbox

sudo /usr/local/bin/init-firewall.sh

exec claude "$@"
