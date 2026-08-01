# Shared environment guard for robocode experiment launchers. Source, do not run.
#
#   source scripts/experiment_env.sh
#
# Agent runs must authenticate as the Claude Max subscription, never as the pay-per-use
# API account. The claude sandbox backend only ever forwards CLAUDE_CODE_OAUTH_TOKEN (or
# a throwaway copy of ~/.claude), so an API key in the environment is not forwarded
# today, but a non-claude backend would forward it and silently bill the wrong account.
# Removing it here makes that impossible regardless of which backend a launcher picks.
unset ANTHROPIC_API_KEY

# Without a stable token the sandbox falls back to copied file credentials and takes an
# exclusive lock for the whole run, which silently serializes parallel sweeps. The token
# lives in ~/.bashrc, which non-interactive shells never source (it returns early for
# them), so read it directly rather than relying on inheritance.
if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -r "$HOME/.bashrc" ]; then
    CLAUDE_CODE_OAUTH_TOKEN=$(
        sed -nE 's/^export CLAUDE_CODE_OAUTH_TOKEN="([^"]+)".*/\1/p' "$HOME/.bashrc" \
            | head -1
    )
    export CLAUDE_CODE_OAUTH_TOKEN
fi

if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    echo "No CLAUDE_CODE_OAUTH_TOKEN; runs would serialize on the credentials lock." >&2
    echo "Generate one with: claude setup-token" >&2
    exit 1
fi
