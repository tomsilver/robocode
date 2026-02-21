# robocode-sandbox — Docker image for running the Claude agent in isolation.
#
# Based on the official Claude Code devcontainer:
#   https://github.com/anthropics/claude-code/tree/main/.devcontainer
#
# Adds Python 3.11 + uv on top of the official node:20 base and installs all
# robocode Python dependencies into /robocode/.venv.
#
# Build from the repository root:
#   bash docker/build.sh
#   # or:
#   docker build -t robocode-sandbox -f docker/Dockerfile .
#
# Runtime bind-mounts (see docker_sandbox.py):
#   -v <sandbox_dir>:/sandbox          writable — agent writes approach.py here
#   -v <repo>/prpl-mono:/robocode/prpl-mono:ro — overrides the stale in-image copy
#
# The prpl-mono submodule is baked in during the build so that `uv sync` can
# create the editable-install .pth entries pointing to /robocode/prpl-mono/*.
# The bind-mount at runtime transparently replaces the source, so no image
# rebuild is needed for prpl-mono code changes — only for PyPI dep changes.

FROM node:20

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

# ---------------------------------------------------------------------------
# System packages
#   - git, sudo, curl: basic dev tools
#   - jq, dnsutils (dig), aggregate, iptables, ipset, iproute2: firewall
#   - python3.11, python3.11-venv, python3.11-dev: Python runtime
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        sudo \
        curl \
        jq \
        iptables \
        ipset \
        iproute2 \
        dnsutils \
        aggregate \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# uv — fast Python package manager
# ---------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin sh

# ---------------------------------------------------------------------------
# npm-global directory (mirrors official devcontainer setup)
# ---------------------------------------------------------------------------
RUN mkdir -p /usr/local/share/npm-global \
    && chown -R node:node /usr/local/share

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH="$PATH:/usr/local/share/npm-global/bin"

# ---------------------------------------------------------------------------
# Claude Code CLI
# ---------------------------------------------------------------------------
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# ---------------------------------------------------------------------------
# Firewall script — node user gets passwordless sudo for this one script only
# (mirrors the official devcontainer setup)
# ---------------------------------------------------------------------------
COPY docker/init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh \
    && echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" \
       > /etc/sudoers.d/node-firewall \
    && chmod 0440 /etc/sudoers.d/node-firewall

# ---------------------------------------------------------------------------
# Claude config directory for the node user
# ---------------------------------------------------------------------------
RUN mkdir -p /home/node/.claude \
    && chown -R node:node /home/node/.claude

# ---------------------------------------------------------------------------
# Python environment — install all robocode deps into /robocode/.venv
#
# Layer ordering for cache efficiency:
#   1. pyproject.toml + uv.lock  (invalidated when PyPI deps change)
#   2. prpl-mono/                (invalidated when submodule structure changes)
#   3. src/                      (invalidated on every robocode source change)
# ---------------------------------------------------------------------------
WORKDIR /robocode

COPY pyproject.toml uv.lock ./
COPY prpl-mono/ prpl-mono/
COPY src/ src/

RUN uv sync --frozen --python python3.11

# ---------------------------------------------------------------------------
# Entrypoint: init firewall → exec claude
# ---------------------------------------------------------------------------
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ---------------------------------------------------------------------------
# Environment (mirrors official devcontainer)
# ---------------------------------------------------------------------------
ENV DEVCONTAINER=true
ENV CLAUDE_CONFIG_DIR=/home/node/.claude
ENV NODE_OPTIONS="--max-old-space-size=4096"

USER node
WORKDIR /sandbox

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
