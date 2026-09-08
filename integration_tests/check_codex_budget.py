"""Check live Codex budgeting using the production strict Apptainer runner.

Usage (real model calls; requires Codex auth and a strict SIF with native usage):
    python integration_tests/check_codex_budget.py --allow-paid --check resume \
        --budget 2 --image robocode-strict-blackbox.sif --output-dir outputs/budget2
    python integration_tests/check_codex_budget.py --allow-paid --check resume \
        --budget 1 --image robocode-strict-blackbox.sif --output-dir outputs/budget1
    python integration_tests/check_codex_budget.py --allow-paid --check cutoff \
        --budget 0.01 --image robocode-strict-blackbox.sif --output-dir outputs/cutoff

Like other integration_tests scripts, this is NOT run by pytest tests/ in CI.
The deterministic, credential-free counterpart is test_codex_budget_integration.py.
No CLI, response, usage ledger, or runner is mocked here. An in-flight response
can overshoot the estimated dollar cap. Artifacts are retained for investigation;
use a fresh output directory. Failures raise instead of silently skipping.
"""

import argparse
import ast
import asyncio
import json
import math
import os
import shutil
from dataclasses import replace
from pathlib import Path

from omegaconf import DictConfig

from robocode.utils.apptainer_sandbox import (
    ApptainerSandboxConfig,
    run_agent_in_apptainer_sandbox,
)
from robocode.utils.backends.codex import CodexBackend
from robocode.utils.rate_limit import run_with_rate_limit_retry


def _tool_call_count(sandbox: Path) -> int:
    """Count actual native tool-call events, not claims in the final answer."""
    count = 0
    for path in (sandbox / ".agent_sessions/codex").rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("type") == "response_item" and event.get("payload", {}).get(
                "type"
            ) in {"function_call", "custom_tool_call"}:
                count += 1
    return count


def _saved_score(path: Path) -> int:
    """Check this literal toy policy without executing model code on the host."""
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    function = next(node for node in functions if node.name == "score")
    statement = function.body[-1]
    assert isinstance(statement, ast.Return) and statement.value is not None
    value = ast.literal_eval(statement.value)
    assert isinstance(value, int)
    return value


def _setup(
    output_dir: Path, image: Path
) -> tuple[ApptainerSandboxConfig, CodexBackend]:
    """Require the same isolated image as the strict experiment campaign."""
    if not image.is_file() or not shutil.which("apptainer"):
        raise RuntimeError("Requires Apptainer and an existing strict SIF")
    assert not os.environ.get("ROBOCODE_CODEX_CMD"), "Live tests require the real CLI"
    backend = CodexBackend(
        DictConfig(
            {
                "input_usd_per_mtok": 4,
                "cached_input_usd_per_mtok": 0.4,
                "output_usd_per_mtok": 20,
            }
        )
    )
    config = ApptainerSandboxConfig(
        sandbox_dir=output_dir / "sandbox",
        output_filename="approach.py",
        model="gpt-5.6-sol",
        blackbox=True,
        blackbox_strict=True,
        strict_sif_path=image.resolve(),
    )
    return config, backend


def _check_resume(
    config: ApptainerSandboxConfig, backend: CodexBackend, budget: float
) -> None:
    """An actual resumed model fixes a defect despite an active wrap-up warning."""
    config = replace(
        config,
        max_budget_usd=budget,
        prompt=(
            "This is phase one of a continuation integration test. For this phase "
            "only, save approach.py containing exactly 'def score():\\n    return 0\\n' "
            "(actual newlines). Then stop immediately WITHOUT creating the confidence "
            "marker. The policy is deliberately incorrect; phase two will validate "
            "and fix it. This intentional phase boundary overrides the instruction "
            "to keep working until confident. Do not delegate."
        ),
    )
    first = asyncio.run(run_agent_in_apptainer_sandbox(config, backend))
    assert first.unconfirmed_solution, first.error
    assert first.total_cost_usd is not None and 0 < first.total_cost_usd < budget - 0.5
    assert _saved_score(config.sandbox_dir / "approach.py") == 0
    previous_tool_calls = _tool_call_count(config.sandbox_dir)
    resume = replace(
        config,
        resume_previous_session=True,
        max_budget_usd=budget - first.total_cost_usd,
        prompt=(
            "Phase two: phase one's instruction to stop immediately no longer applies. "
            "The specification is score() == 1. Use python3 to run the saved policy "
            "and verify the current result. Fix it, then run an assertion of the "
            "specification and save the test command and its result in validation.txt. "
            "Do not delegate. Decide confidence honestly after validation."
        ),
    )
    final = run_with_rate_limit_retry(None, None, backend, apptainer_config=resume)
    assert final.success, final.error
    assert final.generation_metrics is not None
    assert final.generation_metrics.stop_reason is None, final.error
    assert final.total_cost_usd is not None and final.total_cost_usd > 0
    assert final.output_file is not None and _saved_score(final.output_file) == 1
    assert (config.sandbox_dir / "validation.txt").read_text().strip()
    assert _tool_call_count(config.sandbox_dir) > previous_tool_calls
    sessions = config.sandbox_dir / ".agent_sessions/codex"
    assert (sessions / "solution_confident").exists()
    status = json.loads((sessions / "budget_status.json").read_text())
    assert math.isclose(status["max_budget_usd"], budget)
    assert math.isclose(
        status["spent_usd"], first.total_cost_usd + final.total_cost_usd
    )
    assert status["wrap_up"] is True
    print(
        "OK  resumed model used tools, corrected policy and confirmed; "
        f"${status['spent_usd']:.6f}"
    )


def _check_cutoff(
    config: ApptainerSandboxConfig, backend: CodexBackend, budget: float
) -> None:
    """Native real-model usage triggers a cutoff, without confidence or resume."""
    policy = config.sandbox_dir.parent / "seed_policy.py"
    policy.write_text("def score():\n    return 1\n")
    config = replace(
        config,
        init_files={"approach.py": policy},
        max_budget_usd=budget,
        prompt=(
            "Keep approach.py intact. Use your tools to test score() repeatedly "
            "until the external budget monitor stops you. Do not create the "
            "confidence marker and do not delegate."
        ),
    )
    result = run_with_rate_limit_retry(None, None, backend, apptainer_config=config)
    assert result.success, result.error
    assert result.total_cost_usd is not None and result.total_cost_usd >= budget
    assert result.generation_metrics is not None
    assert result.generation_metrics.stop_reason == "error_max_budget_usd"
    assert result.generation_metrics.unconfirmed_solution_retries == 0
    assert result.output_file is not None and _saved_score(result.output_file) == 1
    print(
        f"OK  native usage cutoff at ${result.total_cost_usd:.6f}; saved policy retained"
    )


def main() -> None:
    """Run one explicitly authorized live check, retaining all artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--check", choices=("resume", "cutoff"), required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_paid:
        parser.error("Pass --allow-paid to acknowledge real model usage")
    if not math.isfinite(args.budget) or args.budget <= 0:
        parser.error("Budget must be finite and positive")
    if args.check == "resume" and not 0.5 < args.budget <= 2:
        parser.error("Resume warning check needs a budget above $0.50 and at most $2")
    config, backend = _setup(args.output_dir.resolve(), args.image)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Running {args.check}, max ${args.budget:.2f}; artifacts: {args.output_dir}")
    if args.check == "resume":
        _check_resume(config, backend, args.budget)
    else:
        _check_cutoff(config, backend, args.budget)


if __name__ == "__main__":
    main()
