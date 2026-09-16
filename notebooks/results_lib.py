"""Load RoboCode experiment results and build the ICRA paper tables and figures.

Results are discovered by walking one or more roots for ``results.json`` files
(local folders and/or archives synced from the shared Drive) and parsed from the
Experiment ID directory name ``<env>__<approach>__<prims>__<access>__<model>__timeout_<t>s__<hash>``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- environments

FAMILY_ORDER = [
    "Kinematic 2D",
    "Dynamic 2D",
    "Kinematic 3D",
    "Dynamic 3D",
    "PDDLStream",
]

# env key prefix (before "_generalized"/"__") -> (family, display name)
ENVS: dict[str, tuple[str, str]] = {
    "stickbutton2d": ("Kinematic 2D", "StickButton"),
    "obstruction2d": ("Kinematic 2D", "Obstruction"),
    "clutteredstorage2d": ("Kinematic 2D", "ClutteredStorage"),
    "clutteredretrieval2d": ("Kinematic 2D", "ClutteredRetrieval"),
    "motion2d": ("Kinematic 2D", "Motion"),
    "pushpullhook2d": ("Kinematic 2D", "PushPullHook"),
    "dynobstruction2d": ("Dynamic 2D", "Obstruction"),
    "dynpushpullhook2d": ("Dynamic 2D", "PushPullHook"),
    "dynpusht2d": ("Dynamic 2D", "PushT"),
    "dynscooppour2d": ("Dynamic 2D", "ScoopPour"),
    "obstruction3d": ("Kinematic 3D", "Obstruction"),
    "packing3d": ("Kinematic 3D", "Packing"),
    "transport3d": ("Kinematic 3D", "Transport"),
    "table3d": ("Kinematic 3D", "Table"),
    "shelf3d": ("Kinematic 3D", "Shelf"),
    "balancebeam3d": ("Dynamic 3D", "BalanceBeam"),
    "constrainedcupboard3d": ("Dynamic 3D", "ConstrainedCupboard"),
    "dynamo3d": ("Dynamic 3D", "Dynamo"),
    "rearrange3d": ("Dynamic 3D", "Rearrange"),
    "scooppour3d": ("Dynamic 3D", "ScoopPour"),
    "dynamicshelf3d": ("Dynamic 3D", "Shelf"),
    "sortclutteredblocks3d": ("Dynamic 3D", "SortClutteredBlocks"),
    "sweepintodrawer3d": ("Dynamic 3D", "SweepIntoDrawer"),
    "sweepsimple3d": ("Dynamic 3D", "SweepSimple"),
    "tossing3d": ("Dynamic 3D", "Tossing"),
    "pr2packed": ("PDDLStream", "Packing"),
    "pr2blocked": ("PDDLStream", "Blocked"),
    "rovers": ("PDDLStream", "Rovers"),
}

# Environments for which the benchmark ships no planner: a dash in the planner column.
# kinder-bilevel-planning carries Dynamic 3D models only for the dynamic Shelf, SweepIntoDrawer
# and Tossing tasks (the last on the planner-baselines branch).
NO_PLANNER = {
    "pushpullhook2d",
    "dynpusht2d",
    "dynscooppour2d",
    "obstruction3d",
    "table3d",
    "balancebeam3d",
    "constrainedcupboard3d",
    "dynamo3d",
    "rearrange3d",
    "scooppour3d",
    "sortclutteredblocks3d",
    "sweepsimple3d",
}

# --------------------------------------------------------------------------- conditions

# Paper columns, in table order. Each maps to a predicate over a parsed row.
COLUMNS = ["planner", "genplan", "cc_opus5", "cc_older", "codex", "source"]
COLUMN_LABELS = {
    "planner": "Planner",
    "genplan": "GenPlan",
    "cc_opus5": "Claude Code Opus 5",
    "cc_older": "Claude Code older",
    "codex": "Codex GPT-5.6 Sol",
    "source": "+ source",
}


def condition(row: pd.Series) -> str | None:
    """Map a parsed run to a paper column, or None if it belongs to no column."""
    a, access, model = row["approach"], row["access"], row["model"]
    if a in ("bilevel_planning", "pddlstream_planning"):
        return "planner"
    if a == "llm_genplan":
        return "genplan"
    if a != "agentic":
        return None
    if access == "whitebox" and model == "claude_opus5":
        return "source"
    if access == "blackbox_strict":
        if model == "claude_opus5":
            return "cc_opus5"
        if model.startswith("codex") or "gpt" in model:
            return "codex"
        if model.startswith("claude"):
            return "cc_older"
    return None  # legacy (helper-leaking) black box and anything else


_ENV_SUFFIX_RE = re.compile(r"(_generalized(_[a-z])?|_easy|_medium|_hard|_count\d+)$")


def parse_experiment_id(exp_id: str) -> dict | None:
    """Split an Experiment ID directory name into its fields.

    Layouts: ``env__agentic__prims__access[__strict]__model__timeout_Ts__hash``,
    ``env__llm_genplan__prims__model__timeout_Ts__hash`` and
    ``env__<planner>__prims__timeout_Ts__hash``.
    """
    toks = exp_id.split("__")
    if (
        len(toks) < 5
        or not toks[-2].startswith("timeout_")
        or not re.fullmatch(r"[0-9a-f]{6,}", toks[-1])
    ):
        return None
    env, approach, prims, *mid = toks[:-2]
    access, model = "none", ""
    if approach == "agentic":
        if not mid:
            return None
        access = mid[0]
        rest = mid[1:]
        if rest and rest[0] == "strict":
            access, rest = access + "_strict", rest[1:]
        model = rest[0] if rest else ""
    elif mid:
        model = mid[0]
    return {
        "env": _ENV_SUFFIX_RE.sub("", env),
        "env_key": env,
        "approach": approach,
        "prims": prims,
        "access": access,
        "model": model,
        "timeout": int(toks[-2][len("timeout_") : -1]),
        "hash": toks[-1],
        "experiment_id": exp_id,
    }


# --------------------------------------------------------------------------- loading


@dataclass
class Run:
    path: Path
    meta: dict
    results: dict
    source: str
    rescore: bool = False
    rerun: bool = False


def _find_experiment_dir(path: Path) -> str | None:
    for part in path.parts:
        if "__" in part and parse_experiment_id(part):
            return part
    return None


def load_runs(roots: dict[str, Path] | list[Path]) -> list[Run]:
    """Walk each root for results.json and parse the Experiment ID from its path.

    Within a root, later run directories come first, so a replicate launched again
    (a new timestamp directory) supersedes its earlier copy when the frame is de-duplicated.
    """
    if isinstance(roots, list):
        roots = {p.name: p for p in roots}
    runs: list[Run] = []
    for label, root in roots.items():
        root = Path(root)
        for rp in sorted(root.rglob("results.json"), reverse=True):
            exp_id = _find_experiment_dir(rp.relative_to(root))
            if exp_id is None:
                continue
            try:
                res = json.load(open(rp))
            except (json.JSONDecodeError, OSError):
                continue
            if "solve_rate" not in res:
                continue
            meta = parse_experiment_id(exp_id)
            meta["replicate"] = str(
                res.get("replicate_seed", rp.parent.name.split("_")[-1])
            )
            path_s = str(rp)
            # A re-run from scratch or a re-evaluation on a fixed environment (``rerun_*``,
            # ``rescore_kinder*`` or a ``<timestamp>_rescore`` run directory) replaces the
            # original; a laptop timing re-score does not.
            replacement = (
                "rerun_" in path_s
                or "rescore_kinder" in path_s
                or bool(re.search(r"_rescore/", path_s))
            )
            runs.append(
                Run(
                    rp,
                    meta,
                    res,
                    label,
                    rescore="rescore" in path_s and not replacement,
                    rerun=replacement,
                )
            )
    return runs


def _by_count(res: dict) -> dict[int, dict]:
    """Per-count solve rates, derived from per_episode when results.json has no by_count."""
    if res.get("by_count"):
        return {int(k): v for k, v in res["by_count"].items()}
    out: dict[int, dict] = {}
    for e in res.get("per_episode", []):
        c = int(e.get("object_count", 0))
        d = out.setdefault(c, {"n": 0, "n_solved": 0, "steps": 0})
        d["n"] += 1
        d["n_solved"] += bool(e.get("solved"))
        d["steps"] += e.get("num_steps") or 0
    return {
        c: {
            "n": d["n"],
            "n_solved": d["n_solved"],
            "solve_rate": d["n_solved"] / d["n"],
            "mean_num_steps": d["steps"] / d["n"],
        }
        for c, d in out.items()
    }


def to_frame(runs: list[Run], prefer_rescore: bool = False) -> pd.DataFrame:
    """One row per run with the fields the tables and figures need.

    A replicate re-run from scratch (``rerun_*``) or re-evaluated on a fixed environment
    (``rescore_kinder*`` or a ``<timestamp>_rescore`` run directory) replaces its original
    and inherits its synthesis metrics.
    With ``prefer_rescore`` a laptop timing re-score also replaces its original;
    otherwise those are dropped. Duplicates of the same (experiment, replicate) across
    roots keep the first root listed; within a root the latest run directory wins.
    """
    rows = []
    for r in runs:
        m, res = r.meta, r.results
        fam, name = ENVS.get(m["env"], ("?", m["env"]))
        rows.append(
            {
                **m,
                "family": fam,
                "env_name": name,
                "column": None,
                "solve_rate": res["solve_rate"],
                "n_episodes": res.get("num_evaluated_episodes"),
                "n_crashed": res.get("num_crashed_episodes"),
                "eval_seed": res.get("eval_seed"),
                "mean_steps": res.get("mean_eval_steps"),
                "cost_usd": res.get("agent_cost_usd"),
                "gen_wall_s": res.get("gen_wall_time_s"),
                "gen_turns": res.get("gen_num_turns"),
                "stop": res.get("gen_stop_reason"),
                "timed_out": sum(
                    1
                    for e in res.get("per_episode", [])
                    if e.get("timed_out")
                    or (e.get("num_steps") == 0 and not e.get("crashed"))
                ),
                "by_count": _by_count(res),
                "per_episode": res.get("per_episode", []),
                "source": r.source,
                "rescore": r.rescore,
                "rerun": r.rerun,
                "path": str(r.path),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["column"] = df.apply(condition, axis=1)
    if not prefer_rescore:
        df = df[~df["rescore"]]
    df = df.sort_values(
        ["experiment_id", "replicate", "rerun", "rescore"],
        ascending=[True, True, False, not prefer_rescore],
        kind="stable",
    )
    synth = ["cost_usd", "gen_wall_s", "gen_turns", "stop"]
    firsts = df[~df["rerun"] & ~df["rescore"]].drop_duplicates(
        ["experiment_id", "replicate"]
    )
    firsts = firsts.set_index(["experiment_id", "replicate"])[synth]
    df = df.drop_duplicates(["experiment_id", "replicate"], keep="first")
    for col in synth:
        key = list(zip(df["experiment_id"], df["replicate"]))
        fill = pd.Series([firsts[col].get(k) for k in key], index=df.index)
        df[col] = df[col].where(df[col].notna(), fill)
    df["family"] = pd.Categorical(df["family"], FAMILY_ORDER + ["?"])
    return df.reset_index(drop=True)


def apply_planner_reevaluation(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Use paired outcomes/timings from an audited sweep, preserving original files."""
    import hashlib

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not (root / "full/summary.json").exists():
        raise ValueError("Planner reevaluation is incomplete")
    result = df.copy(deep=True)
    for run in manifest["runs"]:
        mask = (
            result.column.eq("planner")
            & result.env.eq(run["env"])
            & result.replicate.astype(str).eq(str(run["replicate"]))
        )
        if mask.sum() != 1:
            raise ValueError(
                f"Expected one selected source run: {run['env']} {run['replicate']}"
            )
        index = result.index[mask][0]
        source = Path(result.at[index, "path"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != run["sha256"]:
            raise ValueError(f"Source changed since reevaluation: {source}")
        path = root / "full" / f"{run['env']}-{run['replicate']}" / "episodes.jsonl"
        episodes = sorted(
            [json.loads(line) for line in path.read_text().splitlines()],
            key=lambda e: e["episode"],
        )
        original = result.at[index, "per_episode"]
        if len(original) != 100 or [e["episode"] for e in episodes] != list(range(100)):
            raise ValueError(f"Incomplete or duplicated episodes: {path}")
        for old, new in zip(original, episodes):
            if (
                old["seed"],
                old["object_count"],
                bool(old["solved"]),
                run["sha256"],
            ) != (
                new["seed"],
                new["object_count"],
                new["original_solved"],
                new["source_sha256"],
            ):
                raise ValueError(f"Episode identity mismatch: {path}")
            if new["crashed"] or any(
                not np.isfinite(new[k]) or new[k] < 0
                for k in [
                    "planning_time",
                    "execution_time",
                    "env_step_time",
                    "compute_time_s",
                ]
            ):
                raise ValueError(f"Invalid reevaluation: {path}")
            if not np.isclose(
                new["compute_time_s"], new["planning_time"] + new["execution_time"]
            ):
                raise ValueError(f"Timing sum mismatch: {path}")
        result.at[index, "per_episode"] = episodes
        result.at[index, "by_count"] = _by_count({"per_episode": episodes})
        result.at[index, "solve_rate"] = np.mean([e["solved"] for e in episodes])
        result.at[index, "mean_steps"] = np.mean(
            [e["num_steps"] for e in episodes if e["num_steps"] is not None]
        )
        result.at[index, "n_episodes"] = len(episodes)
        result.at[index, "n_crashed"] = 0
        result.at[index, "timed_out"] = sum(not e["plan_found"] for e in episodes)
        result.at[index, "source"] = "local_pddlstream_reevaluation"
        result.at[index, "path"] = str(path)
    return result


def drop_old_dynamic3d(
    runs: list[Run], batch_tokens: tuple[str, ...], roots: tuple[str, ...] = ()
) -> list[Run]:
    """Drop Dynamic 3D runs from the superseded sweeps before de-duplication.

    The early-September Dynamic 3D sweeps ran on defective environments, so only the
    re-run campaign counts. A re-run can share its Experiment ID with the old sweep, so
    the filter works on the loaded runs: a Dynamic 3D run is dropped when its path
    contains one of ``batch_tokens`` or it came from one of the ``roots`` labels.
    """

    def old(r: Run) -> bool:
        fam = ENVS.get(r.meta["env"], ("?", ""))[0]
        return fam == "Dynamic 3D" and (
            any(t in str(r.path) for t in batch_tokens) or r.source in roots
        )

    return [r for r in runs if not old(r)]


def exclude_runs(runs: list[Run], excluded: dict[str, str]) -> list[Run]:
    """Drop the runs whose path contains a key of ``excluded`` ({run directory: reason}).

    For runs whose score measured a harness or account failure rather than the agent. Keys
    name the run directory (``<timestamp>/replicate_<seed>``) so that a later re-launch of
    the same replicate is kept; the reason travels with the entry so the exclusion stays
    auditable.
    """
    return [r for r in runs if not any(k in str(r.path) for k in excluded)]


# --------------------------------------------------------------------------- Drive sync


def sync_drive(
    folder_id: str,
    cache: Path,
    remote: str = "robocode-drive",
    exclude: tuple[str, ...] = ("Outdated",),
) -> Path:
    """Download every .zip under a Drive folder with rclone and extract new ones.

    Returns the directory holding the extracted archives. Archives whose path
    contains any ``exclude`` token are skipped; unchanged archives are not re-fetched.
    """
    cache = Path(cache)
    zips, runs = cache / "zips", cache / "runs"
    zips.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(
        [
            "rclone",
            "lsjson",
            "--recursive",
            "--files-only",
            "--drive-root-folder-id",
            folder_id,
            f"{remote}:",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for entry in json.loads(listing):
        rel = entry["Path"]
        if not rel.endswith(".zip") or any(tok in rel for tok in exclude):
            continue
        local = zips / rel
        if local.exists() and local.stat().st_size == entry["Size"]:
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rclone",
                "copyto",
                "--drive-root-folder-id",
                folder_id,
                f"{remote}:{rel}",
                str(local),
            ],
            check=True,
            capture_output=True,
        )
        target = runs / rel[:-4]
        shutil.rmtree(target, ignore_errors=True)
        try:
            with zipfile.ZipFile(local) as zf:
                zf.extractall(target)
        except zipfile.BadZipFile:
            print(f"skipping corrupt archive: {rel}")
    return runs


# --------------------------------------------------------------------------- tables


def cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample std, min, max and replicate count of solve_rate per (env, column)."""
    g = df.dropna(subset=["column"]).groupby(
        ["family", "env", "env_name", "column"], observed=True
    )["solve_rate"]
    out = g.agg(
        mean="mean",
        sd=lambda s: s.std(ddof=1) if len(s) > 1 else 0.0,
        lo="min",
        hi="max",
        n="count",
    ).reset_index()
    return out


def table1_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1 as a DataFrame of 'mean ± sd (n)' strings, rows ordered like the paper."""
    st = cell_stats(df)
    st["cell"] = st.apply(
        lambda r: f"{r['mean']:.2f} ± {r['sd']:.2f} (n={int(r['n'])})", axis=1
    )
    piv = st.pivot_table(
        index=["family", "env_name"],
        columns="column",
        values="cell",
        aggfunc="first",
        observed=True,
    )
    return piv.reindex(columns=[c for c in COLUMNS if c in piv.columns])


def table1_latex(
    df: pd.DataFrame, min_reps: int = 5, show_partial: bool = False
) -> str:
    """Emit the Table 1 body (rows only) using the paper's ``\\sd{}`` macro.

    Cells with fewer than ``min_reps`` replicates are left blank unless
    ``show_partial``; planner cells for environments without a planner get a dash.
    """
    st = cell_stats(df).set_index(["env", "column"])
    lines = []
    for fam in FAMILY_ORDER:
        envs = [(k, v[1]) for k, v in ENVS.items() if v[0] == fam]
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{fam}}}}} \\\\")
        for key, name in envs:
            cells = []
            for col in COLUMNS:
                if col == "planner" and key in NO_PLANNER:
                    cells.append("--")
                    continue
                if (key, col) in st.index:
                    r = st.loc[(key, col)]
                    if r["n"] >= min_reps or show_partial:
                        cells.append(
                            f"{r['mean']:.2f}\\sd{{{r['sd']:.2f}}}"
                            + ("" if r["n"] >= min_reps else f"$^{{n={int(r['n'])}}}$")
                        )
                        continue
                cells.append("")
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if fam != FAMILY_ORDER[-1]:
            lines.append("\\midrule")
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures

SHORT_NAMES = {
    "ClutteredStorage": "Storage",
    "ClutteredRetrieval": "Retrieval",
    "ConstrainedCupboard": "Cupboard",
    "SortClutteredBlocks": "SortBlocks",
    "SweepIntoDrawer": "SweepDrawer",
    "SweepSimple": "Sweep",
    "PushPullHook": "PushPull",
    "BalanceBeam": "Balance",
}


def table1_latex_transposed(
    df: pd.DataFrame,
    min_reps: int = 5,
    rotate_headers: bool = True,
    columns=None,
    blocks=None,
    stacked: bool = False,
    short_names: bool = False,
    fit: bool = False,
    size: str = "\\scriptsize",
    angle: int = 90,
    colsep: str = "3pt",
    short_labels: bool = False,
    stagger: int = 0,
    header_size: str = "",
    wrap_names: int = 0,
    families=None,
    ranges: bool = False,
    caption: str | None = None,
    label: str = "tab:main-wide",
    bold_best: bool = False,
    bold_best_max: bool = False,
    all_envs: bool = False,
    blank=False,
    single_column: bool = False,
    sublabels: dict[str, str] | None = None,
) -> str:
    """Full-width variant of Table 1: environments as columns grouped by family, methods as rows.

    Returns a complete ``table*`` environment. ``blocks`` groups families into stacked
    sub-tables sharing one font size; ``stacked`` writes the spread as a tiny row under each
    method instead of inline; ``ranges`` reports that spread as ``[min--max]`` over runs
    instead of the std; ``families`` restricts the columns to the listed families;
    ``short_names`` abbreviates long environment names; ``fit`` rescales each block to the
    text width (font sizes then differ between blocks). ``caption`` replaces the default.
    ``bold_best`` bolds the highest mean per environment among the main-setting rows (every
    column except ``source``); ``bold_best_max`` does the same for the max in the range row.
    The ``source`` row is never bolded, since it is a different access setting. ``all_envs``
    lists every environment of the chosen families even without runs, and ``blank`` (True, or
    a collection of family names, environment keys, or ``env:column`` pairs) leaves those
    cells empty: together they produce a placeholder table for pending results. ``single_column`` emits a ``table``
    float scaled to the column width instead of a full-width ``table*``. ``sublabels`` maps a
    column to a tiny second-line label (the model name) placed under the method name in
    stacked mode.
    """
    st = cell_stats(df).set_index(["env", "column"])
    cols = list(columns or COLUMNS)
    fams = [
        (
            fam,
            [
                (k, v[1])
                for k, v in ENVS.items()
                if v[0] == fam and (all_envs or any((k, c) in st.index for c in cols))
            ],
        )
        for fam in FAMILY_ORDER
        if families is None or fam in families
    ]
    main_cols = [c for c in cols if c != "source"]
    best, best_hi = {}, {}
    for k in (k for _, envs in fams for k, _ in envs):
        rows = [
            st.loc[(k, c)]
            for c in main_cols
            if (k, c) in st.index and st.loc[(k, c), "n"] >= min_reps
        ]
        # A single main-setting row has nothing to be compared with, and an all-zero column has
        # no best method, so nothing is bolded in either case.
        best[k] = max(r["mean"] for r in rows) if len(rows) > 1 else None
        best_hi[k] = max(r["hi"] for r in rows) if len(rows) > 1 else None
        if best_hi[k] is not None and best_hi[k] <= 0:
            best[k] = best_hi[k] = None
    fams = [(f, e) for f, e in fams if e]

    def disp(n):
        n = SHORT_NAMES.get(n, n) if short_names else n
        if wrap_names and len(n) > wrap_names:
            # Break a long CamelCase name into two lines at the boundary nearest its middle.
            cuts = [m.start() for m in re.finditer(r"(?<=[a-z])(?=[A-Z])", n)]
            if cuts:
                c = min(cuts, key=lambda i: abs(i - len(n) / 2))
                return f"\\shortstack[l]{{{n[:c]}\\\\{n[c:]}}}"
        return n

    line = 6.5  # pt per staggered header line at scriptsize/tiny

    def head(n, j=0):
        if stagger:
            # Horizontal labels centred on the column, cycling over ``stagger`` baselines.
            lift = (stagger - 1 - j % stagger) * line
            leader = (
                f"\\makebox[0pt][c]{{\\rule[1pt]{{0.3pt}}{{{lift - 1.5:.1f}pt}}}}"
                if lift > 0
                else ""
            )
            return (
                f"\\multicolumn{{1}}{{c}}{{{leader}\\makebox[0pt][c]{{\\raisebox{{{lift:.1f}pt}}"
                f"{{{header_size}{disp(n)}}}}}}}"
            )
        if not rotate_headers:
            return disp(n)
        if angle == 90:
            return f"\\rotatebox{{90}}{{{header_size}{disp(n)}}}"
        # Slanted headers take no width and hang over the columns to the upper right.
        return f"\\multicolumn{{1}}{{l}}{{\\makebox[0pt][l]{{\\rotatebox{{{angle}}}{{{header_size}{disp(n)}}}}}}}"

    labels = (
        {
            "planner": "Planner",
            "genplan": "GenPlan",
            "cc_opus5": "Claude Code",
            "cc_older": "Claude Code",
            "codex": "Codex",
            "source": "CC + source",
        }
        if short_labels
        else {
            "planner": "Planner",
            "genplan": "GenPlan (Opus 5)",
            "cc_opus5": "Claude Code (Opus 5)",
            "cc_older": "Claude Code (older)",
            "codex": "Codex (GPT-5.6 Sol)",
            "source": "Claude Code + source",
        }
    )

    blanked = set(FAMILY_ORDER) if blank is True else set(blank or ())

    def stat(key, col):
        if col == "planner" and key in NO_PLANNER:
            return None
        if ENVS[key][0] in blanked or key in blanked or f"{key}:{col}" in blanked:
            return ""
        if (key, col) in st.index and st.loc[(key, col), "n"] >= min_reps:
            return st.loc[(key, col)]
        return ""

    def fmt_mean(key, col, r):
        m = f"{r['mean']:.2f}"
        return (
            f"\\textbf{{{m}}}"
            if bold_best and col in main_cols and r["mean"] == best[key]
            else m
        )

    def fmt_hi(key, col, r):
        hi = f"{r['hi']:.2f}"
        return (
            f"\\textbf{{{hi}}}"
            if bold_best_max and col in main_cols and r["hi"] == best_hi[key]
            else hi
        )

    def _header_strut(block_fams):
        # Rotated boxes carry their own height; only staggered labels need explicit room.
        return f"\\rule{{0pt}}{{{stagger * line + 3:.0f}pt}}" if stagger else ""

    def block(block_fams):
        keys = [k for _, envs in block_fams for k, _ in envs]
        out = ["\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule"]
        fam_row, cm, start = [], [], 2
        for fam, envs in block_fams:
            fam_row.append(f"\\multicolumn{{{len(envs)}}}{{c}}{{\\textbf{{{fam}}}}}")
            cm.append(f"\\cmidrule(lr){{{start}-{start + len(envs) - 1}}}")
            start += len(envs)
        out += [
            " & " + " & ".join(fam_row) + " \\\\",
            " ".join(cm),
            _header_strut(block_fams)
            + "\\textbf{Method} & "
            + " & ".join(
                head(n, j)
                for j, n in enumerate(n for _, envs in block_fams for _, n in envs)
            )
            + " \\\\",
            "\\midrule",
        ]
        for col in cols:
            if col == "source":
                out.append("\\midrule")
            stats = [(k, stat(k, col)) for k in keys]
            mean = lambda k, r: (
                "--"
                if r is None
                else ("\\phantom{0.00}" if isinstance(r, str) else fmt_mean(k, col, r))
            )
            if stacked:
                # Empty cells keep an invisible range so every column has the same width and the
                # slanted headers stay evenly spaced.
                sd = lambda k, r: (
                    "{\\tiny\\phantom{[0.00--0.00]}}"
                    if r is None or isinstance(r, str)
                    else (
                        f"{{\\tiny [{r['lo']:.2f}--{fmt_hi(k, col, r)}]}}"
                        if ranges
                        else f"{{\\tiny$\\pm${r['sd']:.2f}}}"
                    )
                )
                sub = (
                    f"{{\\tiny {sublabels[col]}}}"
                    if sublabels and col in sublabels
                    else ""
                )
                out.append(
                    labels[col]
                    + " & "
                    + " & ".join(mean(k, r) for k, r in stats)
                    + " \\\\[-2.5pt]"
                )
                out.append(
                    sub + " & " + " & ".join(sd(k, r) for k, r in stats) + " \\\\"
                )
            else:
                full = lambda k, r: (
                    mean(k, r)
                    if r is None or isinstance(r, str)
                    else f"{fmt_mean(k, col, r)}\\sd{{{r['sd']:.2f}}}"
                )
                out.append(
                    labels[col]
                    + " & "
                    + " & ".join(full(k, r) for k, r in stats)
                    + " \\\\"
                )
        out += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(out)

    groups = [[fe for fe in fams if fe[0] in b] for b in blocks] if blocks else [fams]
    spread = "[min--max] across runs" if ranges else "standard deviation"
    cap = (
        f"mean over five synthesis runs, {spread} below"
        if stacked
        else "mean\\sd{std} over five synthesis runs"
    )
    if caption is None:
        caption = f"Success rate $J_{{H,\\tau}}$ on unseen instances, {cap}."
    env, width = (
        ("table", "\\columnwidth") if single_column else ("table*", "\\textwidth")
    )
    out = [
        f"\\begin{{{env}}}[t]",
        "\\centering",
        size,
        f"\\setlength{{\\tabcolsep}}{{{colsep}}}",
    ]
    for i, g in enumerate(groups):
        out += [f"\\resizebox{{{width}}}{{!}}{{%", block(g), "}"] if fit else [block(g)]
        if i < len(groups) - 1:
            out.append("\\vspace{5pt}\n")
    out += [f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{env}}}"]
    return "\n".join(out)
