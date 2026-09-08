#!/usr/bin/env python3
"""Deterministic CLI fixture: no network, credentials, or real model calls."""

import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    """Follow a sandbox-local test plan and emit native-shaped usage records."""
    plan = json.loads(Path("fake_plan.json").read_text(encoding="utf-8"))
    counter = Path(".fake_cli_calls")
    index = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
    counter.write_text(str(index + 1), encoding="utf-8")
    if index >= len(plan):
        raise SystemExit("Unexpected additional Codex invocation")
    step = plan[index]
    prompt = sys.stdin.read()
    instructions = Path("AGENTS.md").read_text(encoding="utf-8")
    sessions = Path(os.environ["CODEX_HOME"]) / "sessions"
    status = json.loads((sessions / "budget_status.json").read_text(encoding="utf-8"))
    with Path("trace.jsonl").open("a", encoding="utf-8") as trace:
        trace.write(
            json.dumps(
                {
                    "index": index,
                    "prompt": prompt,
                    "instructions": instructions,
                    "status": status,
                    "resume": "resume" in sys.argv,
                }
            )
            + "\n"
        )
    if step.get("assert_finishing_contract"):
        assert "not an instruction to exit" in instructions
        assert "targeted validation" in instructions
        if index:
            assert "authorizes useful finishing work" in prompt
            assert "Identify a remaining uncertainty" in prompt
    if "policy" in step:
        Path("approach.py").write_text(step["policy"], encoding="utf-8")
    if step.get("validate"):
        assert runpy.run_path("approach.py")["score"]() == 1
        with Path("validation.txt").open("a", encoding="utf-8") as evidence:
            evidence.write("passed\n")
    print(json.dumps({"type": "thread.started", "thread_id": "fake-root"}), flush=True)
    child = None
    if step.get("child"):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            start_new_session=True,
        )
        Path("child.pid").write_text(str(child.pid), encoding="utf-8")
    for role, usd in (("root", step["cost"]), ("child", step.get("child_cost", 0))):
        if usd <= 0:
            continue
        event = {
            "type": "token_usage_record",
            "payload": {
                "response_id": f"{role}-{index}",
                "usage": {
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": round(usd * 50_000),
                },
            },
        }
        with (sessions / f"{role}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
            # Deliberately reset UI counters every invocation.
            handle.write(
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 1,
                                    "output_tokens": 1,
                                }
                            },
                        },
                    }
                )
                + "\n"
            )
    time.sleep(0.2)  # let the real monitor observe the ledger
    fault = step.get("fault")
    if fault == "delete_ledger":
        (sessions / "root.jsonl").unlink()
    elif fault == "malformed_ledger":
        with (sessions / "root.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("invalid json\n")
    elif fault == "model_change":
        with (sessions / "root.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"type": "turn_context", "payload": {"model": "unexpected-model"}}
                )
                + "\n"
            )
    elif fault == "status_write_failure":
        (sessions / "budget_status.tmp").mkdir()
    if fault or step.get("wait_for_kill"):
        time.sleep(5)
        Path("survived_cutoff").touch()
    if step.get("confident"):
        (sessions / "solution_confident").touch()
    print(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": "Saved policy; finishing this invocation.",
                },
            }
        ),
        flush=True,
    )
    # Stdout usage must not overwrite the authoritative native ledger.
    print(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 1,
                    "cached_input_tokens": 0,
                    "output_tokens": 1,
                },
            }
        ),
        flush=True,
    )
    if child:
        child.terminate()
        child.wait()


if __name__ == "__main__":
    main()
