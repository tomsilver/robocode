"""Plot five relative count difficulties across the reference's ten shared domains.

Run: python scripts/gen_teaser_fig_line_graphs.py
Edit CONFIG or pass --config settings.json (top-level replacements).

Both panels use exactly the same domains, count mapping, and five replicates.
For each domain/replicate/count, take the success fraction and mean computation
across ALL timed attempts, including failures. As in the reference, cap each
attempt at 60 s before reducing it; set time_cap_s=None for uncapped values.
Then average counts equally within a difficulty level, domains equally within
that replicate, and finally the five replicate means. Bands default to +/- one
sample SD. time_statistic='median' changes only the within-count time reducer.
Original Claude success outcomes are retained by default, as in the reference;
its new timing reevaluation outcomes are selectable with program_success_field.

Automatic count mapping uses normalized ordinal count ranks, not absolute counts:
with more than five counts, combine adjacent ranks into the nearest of five bins;
with fewer, fill empty levels using the nearest observed rank (ties to even).
Repeated counts reuse existing measurements; they are not independent new tasks.
Override any domain with five explicit lists in CONFIG['count_groups'], e.g.
'packing3d': [[1], [1], [2], [3], [3]]. Every observed count must be covered.
The mapping is shared by both methods and metrics and exported for review.

Frozen JSON exports contain real per-attempt records. This script does not use
cached figure summaries, run new evaluations, or combine them with newer runs.
Program and planner measurements came from different hardware/protocol versions;
they support a descriptive comparison, not a matched-hardware speedup claim.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
# Edit these settings; count_groups overrides the automatic mapping per domain.
CONFIG = {
    "timing_root": str(ROOT / "Results/robocode-full-timing-2026-09-08"),
    "domains": [
        "stickbutton2d",
        "obstruction2d",
        "clutteredstorage2d",
        "clutteredretrieval2d",
        "motion2d",
        "dynobstruction2d",
        "dynpushpullhook2d",
        "packing3d",
        "transport3d",
        "shelf3d",
    ],
    "difficulty_labels": ["easy", "medium\neasy", "medium", "medium\nhard", "hard"],
    "count_groups": {},
    "replicate_seeds": [24, 42, 222, 424, 444],
    "program_success_field": "original_solved",
    "time_statistic": "mean",
    "time_cap_s": 60.0,
    "output": str(ROOT / "outputs/teaser_line_graphs/teaser"),
    "formats": ["pdf", "png", "svg"],
    "dpi": 300,
    "transparent": False,
    "figsize": [24, 7.2],
    "band": "std",
    "band_multiplier": 1.0,
    "variance_ddof": 1,
    "font_family": "Times New Roman",
    "font_dir": str(ROOT / "outputs/teaser_line_graphs/fonts"),
    "rc": {
        "font.size": 39,
        "axes.titlesize": 51,
        "axes.labelsize": 50,
        "xtick.labelsize": 38,
        "ytick.labelsize": 38,
        "legend.fontsize": 49,
        "axes.labelpad": 16,
        "axes.titlepad": 24,
        "xtick.major.pad": 14,
        "ytick.major.pad": 14,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "axes.unicode_minus": False,
    },
    "line": {"linewidth": 12.2, "markersize": 22, "markeredgewidth": 0.8},
    "band_style": {"alpha": 0.17, "linewidth": 0},
    "grid": {
        "visible": True,
        "axis": "y",
        "alpha": 0.55,
        "linewidth": 1.5,
        "color": "#999999",
    },
    "hide_spines": ["top", "right"],
    "legend": {
        "loc": "upper center",
        "bbox_to_anchor": [0.5, 1.0],
        "ncol": 2,
        "frameon": False,
    },
    "layout": {
        "left": 0.08,
        "right": 0.985,
        "bottom": 0.20,
        "top": 0.60,
        "wspace": 0.3,
    },
    "panels": {
        "success": {
            "title": "Success rate",
            "xlabel": "Difficulty level",
            "ylabel": "",
            "yscale": "linear",
            "ylim": [0, 1.03],
            "yticks": [0, 0.25, 0.5, 0.75, 1],
            "yticklabels": ["0.0", "0.25", "0.5", "0.75", "1.0"],
            "clip_band": [0, 1],
            "xlim": [0.75, 5.25],
        },
        "time": {
            "title": "Computation time",
            "xlabel": "Difficulty level",
            "ylabel": "",
            "unit_label": {
                "s": "(s)",
                "x": 0,
                "y": 1.08,
                "ha": "center",
                "va": "bottom",
                "fontsize": 40,
            },
            "yscale": "linear",
            "ylim": [0, None],
            "lower_range_scale": {
                "breakpoints_s": [0.1, 2.0],
                "height_fractions": [0.05, 0.20],
            },
            "yticks": [0.1, 2, 25, 50],
            "yticklabels": ["0.1", "2", "25", "50"],
            "clip_band": [0, None],
            "xlim": [0.75, 5.25],
        },
    },
    "show_coverage": False,
    "show_xticks": False,
    "caption": "",
    "caption_style": {
        "x": 0.5,
        "y": 0.035,
        "ha": "center",
        "va": "bottom",
        "fontsize": 36,
    },
    "methods": {
        "program": {
            "label": "Claude Code Opus 5",
            "color": "#276FBF",
            "marker": "o",
            "linestyle": "-",
        },
        "planner": {
            "label": "Planner",
            "color": "#D46A36",
            "marker": "s",
            "linestyle": "--",
        },
    },
}


def map_counts(counts):
    """Map observed count ranks to five bins, repeating nearest ranks if needed."""
    groups = [[] for _ in range(5)]
    slots = np.rint(np.linspace(0, 4, len(counts))).astype(int)
    for count, slot in zip(counts, slots):
        groups[slot].append(count)
    for level, group in enumerate(groups):
        if not group:
            group.append(counts[int(np.rint(level * (len(counts) - 1) / 4))])
    return groups


def main():
    """Aggregate real episode records and export plots, mapping, and statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="JSON overrides of CONFIG top-level keys"
    )
    parser.add_argument("--output", type=Path, help="Output prefix without extension")
    args = parser.parse_args()
    cfg = copy.deepcopy(CONFIG)
    if args.config:
        overrides = json.loads(args.config.read_text())
        if unknown := overrides.keys() - cfg.keys():
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        cfg.update(overrides)
    if args.output:
        cfg["output"] = str(args.output)
    domains, seeds = cfg["domains"], cfg["replicate_seeds"]
    methods = list(cfg["methods"])
    if len(domains) != 10 or len(set(domains)) != 10:
        raise ValueError("Exactly the ten runtime-comparison domains are required")
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five distinct replicate seeds are required")
    if len(cfg["difficulty_labels"]) != 5:
        raise ValueError("Exactly five difficulty labels are required")
    if set(methods) != {"program", "planner"}:
        raise ValueError("Keep program/planner method keys; edit labels/styles instead")
    if cfg["program_success_field"] not in {"original_solved", "solved"}:
        raise ValueError("program_success_field must be original_solved or solved")
    if cfg["time_statistic"] not in {"mean", "median"}:
        raise ValueError("time_statistic must be mean or median")
    if cfg["band"] not in {"std", "sem", "none"} or cfg["variance_ddof"] not in {0, 1}:
        raise ValueError("Use band std/sem/none and variance_ddof 0/1")
    if cfg["time_cap_s"] is not None and cfg["time_cap_s"] <= 0:
        raise ValueError("time_cap_s must be positive or null")
    if set(cfg["count_groups"]) - set(domains):
        raise ValueError("count_groups contains an unknown domain")

    root = Path(cfg["timing_root"])
    frames, sources = {}, []
    for method, filename in [
        ("program", "timed-episodes.json"),
        ("planner", "planner-episodes.json"),
    ]:
        path = root / filename
        raw = path.read_bytes()
        records = json.loads(raw)  # Preserve exact 63-bit task seeds.
        if method == "planner" and {r["env"] for r in records} != set(domains):
            raise ValueError("Configured domains must match the ten planner domains")
        frames[method] = [r for r in records if r["env"] in domains]
        sources.append(
            {
                "method": method,
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    run_groups, cells, identities = {}, {}, {}
    missing = dict.fromkeys(methods, 0)
    for method, records in frames.items():
        if len(records) != 5000:
            raise ValueError(
                f"{method}: expected 10 domains x 5 replicates x 100 tasks"
            )
        identities[method] = set()
        for episode in records:
            env, seed, count = (
                episode["env"],
                episode["replicate"],
                episode["object_count"],
            )
            if seed not in seeds or count is None or int(count) != count:
                raise ValueError(
                    f"Invalid replicate/count: {method}, {env}, {seed}, {count}"
                )
            if type(episode["solved"]) is not bool:
                raise ValueError("Every task must have a Boolean outcome")
            success_key = (
                cfg["program_success_field"] if method == "program" else "solved"
            )
            if type(episode[success_key]) is not bool:
                raise ValueError(f"Missing Boolean {success_key}")
            identity = (env, seed, episode["seed"], int(count))
            if identity in identities[method]:
                raise ValueError(f"Duplicate task: {method}, {identity}")
            identities[method].add(identity)
            run_groups.setdefault((method, env, seed), []).append(episode)
            cells.setdefault((method, env, seed, int(count)), []).append(episode)
            time_key = "policy_time_s" if method == "program" else "compute_time_s"
            value = episode[time_key]
            if value is None:
                if method != "planner" or not episode.get("crashed"):
                    raise ValueError(f"Unexpected missing time: {method}, {identity}")
                missing[method] += 1
            elif not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid time: {method}, {identity}")
    if identities["program"] != identities["planner"]:
        raise ValueError(
            "Methods must evaluate identical domain/replicate/task-seed/count identities"
        )
    expected_runs = {
        (method, env, seed) for method in methods for env in domains for seed in seeds
    }
    if set(run_groups) != expected_runs or any(
        len(r) != 100 for r in run_groups.values()
    ):
        raise ValueError("Each method/domain needs five 100-task runs")

    mapping, mapping_rows = {}, []
    for env in domains:
        counts = sorted({key[3] for key in cells if key[1] == env})
        groups = cfg["count_groups"].get(env, map_counts(counts))
        if len(groups) != 5 or any(not group for group in groups):
            raise ValueError(f"{env}: provide five nonempty count groups")
        if any(
            len(group) != len(set(group)) or group != sorted(group) for group in groups
        ):
            raise ValueError(
                f"{env}: each count group must be sorted without duplicates"
            )
        if {count for group in groups for count in group} != set(counts):
            raise ValueError(
                f"{env}: every observed count {counts} must be covered, with no extra counts"
            )
        if any(max(left) > min(right) for left, right in zip(groups, groups[1:])):
            raise ValueError(
                f"{env}: difficulty groups must follow increasing count order; repeats are allowed"
            )
        mapping[env] = groups
        for i, group in enumerate(groups):
            mapping_rows.append(
                {
                    "domain": env,
                    "difficulty": i + 1,
                    "counts": " ".join(map(str, group)),
                    "reused_counts": " ".join(
                        str(c) for c in group if sum(c in g for g in groups) > 1
                    ),
                }
            )
        print(f"{env}: {groups}")

    # First reduce individual attempts within each domain/replicate/count.
    per_count = {}
    for (method, env, seed, count), episodes in cells.items():
        key = "policy_time_s" if method == "program" else "compute_time_s"
        times = np.asarray(
            [e[key] for e in episodes if e[key] is not None], dtype=float
        )
        if not len(times):
            raise ValueError(f"No measured attempts: {method}, {env}, {seed}, {count}")
        if cfg["time_cap_s"] is not None:
            times = np.minimum(times, cfg["time_cap_s"])
        success_key = cfg["program_success_field"] if method == "program" else "solved"
        per_count[(method, env, seed, count)] = {
            "success": float(np.mean([e[success_key] for e in episodes])),
            "time": float(getattr(np, cfg["time_statistic"])(times)),
            "attempts": len(episodes),
            "timed_attempts": len(times),
        }
    domain_rows, seed_rows, summary, curves = [], [], [], {}
    for metric in ["success", "time"]:
        for method in methods:
            means, spreads = [], []
            for level in range(5):
                values = []
                for seed in seeds:
                    domain_values = []
                    for env in domains:
                        selected = [
                            per_count[(method, env, seed, count)]
                            for count in mapping[env][level]
                        ]
                        value = float(np.mean([row[metric] for row in selected]))
                        domain_values.append(value)
                        domain_rows.append(
                            {
                                "metric": metric,
                                "method": method,
                                "domain": env,
                                "difficulty": level + 1,
                                "replicate_seed": seed,
                                "value": value,
                                "counts": " ".join(map(str, mapping[env][level])),
                                "attempts": sum(r["attempts"] for r in selected),
                                "timed_attempts": sum(
                                    r["timed_attempts"] for r in selected
                                ),
                            }
                        )
                    value = float(np.mean(domain_values))
                    values.append(value)
                    seed_rows.append(
                        {
                            "metric": metric,
                            "method": method,
                            "difficulty": level + 1,
                            "replicate_seed": seed,
                            "value": value,
                            "domains": len(domains),
                        }
                    )
                mean = float(np.mean(values))
                variance = float(np.var(values, ddof=cfg["variance_ddof"]))
                spread = np.sqrt(variance)
                if cfg["band"] == "sem":
                    spread /= np.sqrt(len(seeds))
                elif cfg["band"] == "none":
                    spread = 0.0
                spread *= cfg["band_multiplier"]
                means.append(mean)
                spreads.append(spread)
                summary.append(
                    {
                        "metric": metric,
                        "method": method,
                        "difficulty": level + 1,
                        "mean": mean,
                        "variance": variance,
                        "band_halfwidth": float(spread),
                        "seeds": len(seeds),
                        "domains": len(domains),
                    }
                )
            curves[(metric, method)] = (np.asarray(means), np.asarray(spreads))

    font_dir = Path(cfg["font_dir"]) if cfg["font_dir"] else None
    if font_dir and font_dir.is_dir():
        for path in sorted(font_dir.iterdir()):
            if path.suffix.lower() in {".ttf", ".otf"}:
                font_manager.fontManager.addfont(str(path))
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=cfg["font_family"]),
        fallback_to_default=False,
    )
    plt.rcParams.update(cfg["rc"])
    plt.rcParams["font.family"] = cfg["font_family"]
    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    fig.subplots_adjust(**cfg["layout"])
    x = np.arange(1, 6)
    for ax, metric in zip(axes, ["success", "time"]):
        panel = cfg["panels"][metric]
        for method in methods:
            mean, spread = curves[(metric, method)]
            style = cfg["methods"][method]
            ax.plot(x, mean, **cfg["line"], **style)
            lower, upper = mean - spread, mean + spread
            lo, hi = panel["clip_band"]
            if lo is not None:
                lower = np.maximum(lower, lo)
            if hi is not None:
                upper = np.minimum(upper, hi)
            if panel["yscale"] == "log":
                lower = np.where(lower > 0, lower, np.nan)
            if cfg["band"] != "none":
                ax.fill_between(
                    x, lower, upper, color=style["color"], **cfg["band_style"]
                )
        labels = cfg["difficulty_labels"]
        if cfg["show_coverage"]:
            labels = [f"{label}\n10 domains" for label in labels]
        if cfg["show_xticks"]:
            ax.set_xticks(x, labels)
        else:
            ax.set_xticks([])
        ax.set_xlim(*panel["xlim"])
        ax.set(
            title=panel["title"],
            xlabel=panel["xlabel"],
            ylabel=panel["ylabel"],
            yscale=panel["yscale"],
        )
        if panel["ylim"] is not None:
            ax.set_ylim(*panel["ylim"])
        if panel["yticks"] is not None:
            ax.set_yticks(panel["yticks"], labels=panel.get("yticklabels"))
        if metric == "time" and panel.get("lower_range_scale"):
            # Stretch only the displayed coordinates; values and SDs stay in seconds.
            if panel["yscale"] != "linear":
                raise ValueError("lower_range_scale requires a linear base yscale")
            ymin, ymax = ax.get_ylim()
            seconds = np.array(
                [ymin, *panel["lower_range_scale"]["breakpoints_s"], ymax]
            )
            heights = np.array([0, *panel["lower_range_scale"]["height_fractions"], 1])
            if (
                len(seconds) != len(heights)
                or not np.all(np.diff(seconds) > 0)
                or not np.all(np.diff(heights) > 0)
            ):
                raise ValueError(
                    "Scale breakpoints and fractions must match, increase, "
                    "and lie inside ylim and (0, 1), respectively"
                )

            def forward(values):
                """Map seconds to a piecewise-linear axis coordinate."""
                values = np.asarray(values)
                i = np.clip(np.searchsorted(seconds, values) - 1, 0, len(seconds) - 2)
                return heights[i] + (values - seconds[i]) * (
                    heights[i + 1] - heights[i]
                ) / (seconds[i + 1] - seconds[i])

            def inverse(values):
                """Map the displayed coordinate back to seconds."""
                values = np.asarray(values)
                i = np.clip(np.searchsorted(heights, values) - 1, 0, len(heights) - 2)
                return seconds[i] + (values - heights[i]) * (
                    seconds[i + 1] - seconds[i]
                ) / (heights[i + 1] - heights[i])

            ax.set_yscale("function", functions=(forward, inverse))
            ax.set_ylim(ymin, ymax)
            if panel["yticks"] is not None:
                ax.set_yticks(panel["yticks"], labels=panel.get("yticklabels"))
        if "unit_label" in panel:
            ax.text(transform=ax.transAxes, **panel["unit_label"])
        ax.grid(**cfg["grid"])
        for spine in cfg["hide_spines"]:
            ax.spines[spine].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), **cfg["legend"])
    band = (
        f"+/- {cfg['band_multiplier']:g} {'SD' if cfg['band'] == 'std' else 'SEM'}"
        if cfg["band"] != "none"
        else "none"
    )
    cap = (
        f"(capped at {cfg['time_cap_s']:g} s)"
        if cfg["time_cap_s"] is not None
        else "(uncapped)"
    )
    success = (
        "Original"
        if cfg["program_success_field"] == "original_solved"
        else "Reevaluated Claude / original Planner"
    )
    if cfg["caption"]:
        fig.text(
            s=cfg["caption"].format(band=band, cap=cap, success=success),
            **cfg["caption_style"],
        )
    output = Path(cfg["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in cfg["formats"]:
        path = Path(f"{output}.{suffix}")
        fig.savefig(path, dpi=cfg["dpi"], transparent=cfg["transparent"])
        print(f"Saved {path}")
    plt.close(fig)
    for suffix, records in [
        ("summary", summary),
        ("seed_means", seed_rows),
        ("domain_means", domain_rows),
        ("count_mapping", mapping_rows),
    ]:
        with Path(f"{output}_{suffix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    audit = {
        "config": cfg,
        "sources": sources,
        "resolved_font": font_path,
        "count_mapping": mapping,
        "attempts_per_method": {m: len(f) for m, f in frames.items()},
        "untimed_attempts_excluded_from_time_only": missing,
        "claude_changed_outcomes": sum(
            e["solved"] != e["original_solved"] for e in frames["program"]
        ),
        "aggregation": "Per count/seed success fraction and time reducer; equal counts within level, equal domains within seed, then mean and variance across five seeds. Reused counts are correlated across levels.",
        "provenance": json.loads((root / "provenance.json").read_text()),
    }
    Path(f"{output}_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    print(
        "Read 10,000 real attempts: 10 shared domains x 5 seeds x 100 tasks x 2 methods."
    )
    print(
        f"Every point includes all ten domains and all five seeds. Untimed attempts: {missing}"
    )


if __name__ == "__main__":
    main()
