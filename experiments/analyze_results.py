"""Load and display experiment results from one or more Hydra run directories."""

import argparse
import json
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf


def collect_results(search_dirs: list[Path]) -> pd.DataFrame:
    """Recursively find results.json files and build a DataFrame."""
    rows: list[dict] = []
    for search_dir in search_dirs:
        for results_path in sorted(search_dir.rglob("results.json")):
            job_dir = results_path.parent
            config_path = job_dir / ".hydra" / "config.yaml"
            overrides_path = job_dir / ".hydra" / "overrides.yaml"
            if not config_path.exists():
                continue

            with open(results_path, encoding="utf-8") as results_file:
                results = json.load(results_file)

            cfg = OmegaConf.load(config_path)
            assert isinstance(cfg, DictConfig)

            row: dict = {}

            if overrides_path.exists():
                overrides = OmegaConf.load(overrides_path)
                assert isinstance(overrides, ListConfig)
                for override in overrides:
                    assert isinstance(override, str)
                    key, val = override.split("=", 1)
                    if key == "seed":
                        key = "replicate_seed"
                    row[key] = val

            if "approach" not in row:
                row["approach"] = cfg["approach"]["_target_"].rsplit(".", 1)[-1]
            if "environment" not in row:
                row["environment"] = cfg["environment"]["_target_"].rsplit(".", 1)[-1]
            if "replicate_seed" not in row:
                row["replicate_seed"] = cfg.get("replicate_seed", cfg.get("seed"))

            for key, value in results.items():
                # These nested structures stay in results.json; their headline
                # metrics are surfaced through separate flat fields below.
                if key in ("per_episode", "gen_model_usage", "count_regimes"):
                    continue
                # by_count is a nested {count: {...}} dict; flatten its solve rates to
                # numeric solve_rate@<count> columns so they aggregate across seeds.
                if key == "by_count" and isinstance(value, dict):
                    for count, entry in value.items():
                        if isinstance(entry, dict) and "solve_rate" in entry:
                            row[f"solve_rate@{count}"] = entry["solve_rate"]
                    continue
                # Booleans are per-run flags (eval_complete, gen_turn_limit_hit, ...).
                # Left as bools they are not a numeric dtype, so aggregate_results
                # treats them as *grouping* keys and splits one condition into a row
                # per flag value -- replicates that differ only in whether a limit was
                # hit stop being averaged together. As floats they aggregate into the
                # fraction of replicates where the flag held, which is what a summary
                # table wants: eval_complete 0.8 means one replicate in five is partial.
                row[key] = float(value) if isinstance(value, bool) else value
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Average replicate metrics without dropping rows with optional metadata."""
    numeric_cols = dataframe.select_dtypes(include="number").columns.tolist()
    if "replicate_seed" in numeric_cols:
        numeric_cols.remove("replicate_seed")
    exclude_from_group = {"replicate_seed", "approach.load_dir"}
    group_cols = [
        c
        for c in dataframe.columns
        if c not in numeric_cols and c not in exclude_from_group
    ]

    seed_info = (
        dataframe.groupby(group_cols, sort=False, dropna=False)["replicate_seed"]
        .agg(replicate_seeds=lambda s: sorted(s.astype(str)))
        .reset_index()
    )
    seed_info["replicate_seeds"] = seed_info["replicate_seeds"].apply(",".join)
    seed_info["n_replicates"] = seed_info["replicate_seeds"].str.count(",") + 1

    averaged = (
        dataframe.drop(columns=["replicate_seed"])
        .groupby(group_cols, sort=False, dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    averaged = averaged.merge(seed_info, on=group_cols)

    col_order = group_cols + ["n_replicates", "replicate_seeds"] + numeric_cols
    averaged = averaged[[c for c in col_order if c in averaged.columns]]

    sort_cols = ["environment"] + [c for c in group_cols if c != "environment"]
    return averaged.sort_values(sort_cols).reset_index(drop=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="Hydra output directories to scan for results.json files.",
    )
    args = parser.parse_args()
    dataframe = collect_results(args.dirs)
    if dataframe.empty:
        print("No results found.")
        return

    averaged = aggregate_results(dataframe)

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 200
    ):
        print(averaged.to_string(index=False))


if __name__ == "__main__":
    _main()
