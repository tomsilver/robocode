"""The standalone live check must never spend without an explicit opt-in."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "integration_tests/check_codex_budget.py"


def test_live_budget_check_help_is_unpaid():
    """Help imports successfully without an image, credentials or model calls."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--allow-paid" in result.stdout


@pytest.mark.parametrize(
    "extra,error",
    [
        ([], "Pass --allow-paid"),
        (["--allow-paid", "--budget", "nan"], "finite and positive"),
        (["--allow-paid", "--budget", "0"], "finite and positive"),
        (["--allow-paid", "--budget", "0.5"], "Resume warning check"),
        (["--allow-paid", "--budget", "3"], "Resume warning check"),
    ],
)
def test_live_budget_check_rejects_unsafe_launch(tmp_path, extra, error):
    """Argument validation happens before auth, image access or model launch."""
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "resume",
            "--budget",
            "2",
            "--image",
            str(tmp_path / "absent.sif"),
            "--output-dir",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert error in result.stderr
    assert not output.exists()
