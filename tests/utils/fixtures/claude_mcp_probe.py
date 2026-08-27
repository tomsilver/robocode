"""Run the real Claude completion wrapper inside the GenPlan Docker sandbox.

Both tests use this program. --without-deny-flag reproduces the old wrapper;
otherwise all production restrictions remain. Each call has a $1 CLI budget.
"""

import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from omegaconf import DictConfig

from robocode.utils.llm.cli_client import ClaudeCLIClient


def main() -> None:
    """Call Claude with the test MCP server and save the full CLI transcript."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-deny-flag", action="store_true")
    options = parser.parse_args()
    real_run = subprocess.run

    def record_cli(args, **kwargs):
        args = list(args)
        if options.without_deny_flag:
            # The control differs only by removing the production deny-all flag.
            index = args.index("--disallowedTools")
            del args[index : index + 2]
        args[args.index("--output-format") + 1] = "stream-json"
        args.extend(
            [
                "--verbose",
                "--strict-mcp-config",
                "--mcp-config",
                "/sandbox/mcp.json",
                "--allowedTools",
                "mcp__probe__probe_token",
                "--setting-sources",
                "",
                "--no-session-persistence",
                "--max-budget-usd",
                "1.0",
            ]
        )
        result = real_run(args, check=kwargs.pop("check", True), **kwargs)
        Path("/sandbox/transcript.jsonl").write_text(result.stdout, encoding="utf-8")
        events = [
            json.loads(line) for line in result.stdout.splitlines() if line.strip()
        ]
        final = next(e for e in reversed(events) if e.get("type") == "result")
        assert not final.get("is_error"), final
        result.stdout = json.dumps(final)
        return result

    client = ClaudeCLIClient(
        DictConfig({"model": "claude-opus-5", "request_timeout_s": 120})
    )
    with patch("robocode.utils.llm.cli_client.subprocess.run", record_cli):
        client.complete(
            [
                {
                    "role": "user",
                    "content": (
                        "Call mcp__probe__probe_token and return its token exactly. "
                        "Do not guess. If unavailable, respond UNAVAILABLE."
                    ),
                }
            ]
        )


if __name__ == "__main__":
    main()
