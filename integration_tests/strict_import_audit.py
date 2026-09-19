"""Audit strict-image imports, including deliberate access to the MCP environment.

Run from the repository root. Exit 1 means packages beyond NumPy/SciPy are
reachable by the agent, even if domain/simulator packages remain absent.
"""

import argparse
import json
import subprocess
from pathlib import Path

from robocode.utils.apptainer_environment import clean_apptainer_env
from robocode.utils.apptainer_sandbox import (
    ApptainerSandboxConfig,
    _build_apptainer_cmd,
)

DOMAIN_MODULES = (
    "robocode.environments",
    "robocode.primitives",
    "kinder",
    "gymnasium",
    "shapely",
    "pybullet",
)
EXTRA_MODULES = (
    "robocode",
    "mcp",
    "pydantic",
    "httpx",
    "pip",
    "setuptools",
    "mercurial",
    "packaging",
    "gyp",
    "codegen",
    "libstdcxx",
    "debpython",
)
PAYLOAD = """import importlib,json,sys,socket
from pathlib import Path
if len(sys.argv)>1:
    sys.path[:0] = ['/opt/robocode-mcp/lib/python3.11/site-packages', '/usr/lib/python3/dist-packages', '/usr/local/lib/python3.11/dist-packages', '/usr/local/lib/node_modules/npm/node_modules/node-gyp/gyp/pylib', '/usr/share/glib-2.0', '/usr/share/gcc/python', '/usr/share/python3']
result={}
for name in ['numpy','scipy','robocode','robocode.environments','robocode.primitives','kinder','gymnasium','shapely','pybullet','mcp','pydantic','httpx','pip','setuptools','mercurial','packaging','gyp','codegen','libstdcxx','debpython']:
    try:
        module=importlib.import_module(name)
        result[name]={'imported':True,'file':getattr(module,'__file__',None)}
    except Exception as exc:
        result[name]={'imported':False,'error':str(exc)}
assert {name for _,name in socket.if_nameindex()} == {'lo'}
allowed = [Path('/usr/lib/python3.11'), Path('/opt/robocode-strict/lib/python3.11/site-packages/numpy'), Path('/opt/robocode-strict/lib/python3.11/site-packages/scipy')]
foreign = [str(p) for root in [Path('/usr'), Path('/opt')] for p in root.rglob('__init__.py') if not any(p.is_relative_to(a) for a in allowed)]
print(json.dumps({'executable':sys.executable,'modules':result,'foreign_package_sources':foreign}))
"""


def audit(results: Path, image: Path) -> dict:
    """Test the actual image rather than relying on its build recipe or package list."""
    results = results.resolve()
    results.mkdir(parents=True, exist_ok=False)
    (results / "probe.py").write_text(PAYLOAD, encoding="utf-8")
    config = ApptainerSandboxConfig(
        sandbox_dir=results, blackbox=True, blackbox_strict=True, strict_sif_path=image
    )
    reports = {}
    for label, python, args in (
        ("strict", "/opt/robocode-strict/bin/python", []),
        ("system", "/usr/bin/python3", []),
        ("strict_with_mcp_path", "/opt/robocode-strict/bin/python", ["add-mcp-path"]),
    ):
        cmd = _build_apptainer_cmd(
            config, str(results), None, None, None, [python, "/sandbox/probe.py", *args]
        )
        proc = subprocess.run(
            cmd,
            env=clean_apptainer_env(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        (results / (label + ".stdout")).write_text(proc.stdout, encoding="utf-8")
        (results / (label + ".stderr")).write_text(proc.stderr, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"Interpreter probe failed: {label}; inspect {results}")
        reports[label] = json.loads(proc.stdout)
    report = {
        "interpreters": reports,
        "domain_modules_hidden": all(
            not r["modules"][m]["imported"]
            for r in reports.values()
            for m in DOMAIN_MODULES
        ),
        "no_foreign_package_sources": all(
            not r["foreign_package_sources"] for r in reports.values()
        ),
        "only_allowed_packages_reachable": all(
            not r["modules"][m]["imported"]
            for r in reports.values()
            for m in EXTRA_MODULES
        ),
    }
    (results / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Return a failed audit when the literal strict package boundary does not hold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--image", type=Path, default=Path.cwd() / "robocode-strict-blackbox.sif"
    )
    args = parser.parse_args()
    report = audit(args.results_dir, args.image)
    print(json.dumps(report, indent=2))
    if (
        not report["domain_modules_hidden"]
        or not report["only_allowed_packages_reachable"]
        or not report["no_foreign_package_sources"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
