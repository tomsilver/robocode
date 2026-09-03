"""Tests for the strict blackbox import allowlist and generated-module isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium.spaces import Box

from robocode.utils.episode import load_generated_approach
from robocode.utils.strict_blackbox import (
    STRICT_ALLOWED_PACKAGES,
    StrictImportError,
    check_strict_imports,
    strict_allowlist_description,
)


def test_strict_allowlist_description_names_every_allowed_package() -> None:
    """The prose the prompt and the rejection share tracks the allowlist constant."""
    description = strict_allowlist_description()
    for package in STRICT_ALLOWED_PACKAGES:
        assert f"`{package}`" in description
    assert "`sys.modules`" in description


def _write(root: Path, name: str, source: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_strict_imports_accept_stdlib_allowed_packages_and_siblings(
    tmp_path: Path,
) -> None:
    """Stdlib, numpy/scipy, plain and relative sibling imports all pass."""
    _write(tmp_path, "helper.py", "import math\nfrom scipy import optimize\nK = 1\n")
    _write(tmp_path, "pkg/__init__.py", "from . import inner\n")
    _write(tmp_path, "pkg/inner.py", "import json\n")
    entry = _write(
        tmp_path,
        "approach.py",
        "import numpy as np\nimport helper\nfrom pkg import inner\n"
        "class GeneratedApproach:\n    pass\n",
    )
    assert check_strict_imports(entry) == {"numpy", "scipy"}


@pytest.mark.parametrize(
    ("files", "offender"),
    [
        ({"approach.py": "import shapely\n"}, "approach.py:1: import shapely"),
        (
            {
                "approach.py": "def f():\n    try:\n        import pybullet\n"
                "    except ImportError:\n        pass\n"
            },
            "approach.py:3: import pybullet",
        ),
        (
            {"approach.py": "import helper\n", "helper.py": "from kinder import x\n"},
            "helper.py:1: import kinder",
        ),
        (
            {"approach.py": "import helper, robocode\n", "helper.py": "K = 1\n"},
            "approach.py:1: import robocode",
        ),
        ({"approach.py": "import importlib\n"}, "import importlib"),
        ({"approach.py": "m = __import__('os')\n"}, "approach.py:1: __import__"),
        (
            {"approach.py": "import builtins\nbuiltins.__import__('os')\n"},
            "approach.py:2: __import__",
        ),
        (
            {"approach.py": "import sys\nenv = sys.modules['robocode']\n"},
            "approach.py:2: sys.modules",
        ),
        (
            {"approach.py": "import sys as s\ns.modules.get('robocode')\n"},
            "approach.py:2: sys.modules",
        ),
        ({"approach.py": "from sys import modules\n"}, "approach.py:1: sys.modules"),
        ({"approach.py": "from .. import helper\n"}, "relative import above"),
        ({"approach.py": "from env_client import make_env\n"}, "import env_client"),
    ],
)
def test_strict_imports_reject_everything_else(
    tmp_path: Path, files: dict[str, str], offender: str
) -> None:
    """Nested, sibling-transitive, dynamic, run-time, and client imports are named."""
    for name, source in files.items():
        _write(tmp_path, name, source)
    with pytest.raises(StrictImportError, match=offender):
        check_strict_imports(tmp_path / "approach.py")


def test_loader_runs_the_strict_check_before_exec(tmp_path: Path) -> None:
    """A strict program with a host-only import never gets executed."""
    entry = _write(
        tmp_path,
        "approach.py",
        "import robocode\nraise SystemExit('must not run')\n",
    )
    space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    with pytest.raises(StrictImportError):
        load_generated_approach(entry, space, space, {}, strict_imports=True)


def test_strict_loader_rejects_sibling_that_shadows_cached_host_module(
    tmp_path: Path,
) -> None:
    """A sibling cannot disguise a host module already present in sys.modules."""
    _write(tmp_path, "robocode.py", "LOCAL = True\n")
    entry = _write(
        tmp_path,
        "approach.py",
        "import robocode\n"
        "class GeneratedApproach:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        assert robocode.LOCAL\n",
    )
    space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    with pytest.raises(StrictImportError, match="cached host module robocode"):
        load_generated_approach(entry, space, space, {}, strict_imports=True)


def test_strict_loader_isolates_same_named_siblings_between_sandboxes(
    tmp_path: Path,
) -> None:
    """Sequential policies may reuse a sibling name without sharing its module."""
    entries: list[Path] = []
    for sandbox_name, value in (("first", 1), ("second", 2)):
        sandbox = tmp_path / sandbox_name
        _write(sandbox, "shared_policy_helper.py", f"VALUE = {value}\n")
        entries.append(
            _write(
                sandbox,
                "approach.py",
                "import shared_policy_helper\n"
                "class GeneratedApproach:\n"
                "    def __init__(self, *args, **kwargs):\n"
                "        self.value = shared_policy_helper.VALUE\n",
            )
        )

    space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    try:
        first = load_generated_approach(
            entries[0], space, space, {}, strict_imports=True
        )
        second = load_generated_approach(
            entries[1], space, space, {}, strict_imports=True
        )

        assert first.value == 1
        assert second.value == 2
        helper = sys.modules["shared_policy_helper"]
        assert helper.__file__ is not None
        assert Path(helper.__file__).is_relative_to(tmp_path / "second")
    finally:
        sys.modules.pop("shared_policy_helper", None)
