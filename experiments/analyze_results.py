"""Load and display experiment results from one or more Hydra run directories."""

import argparse
import json
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf


def _collect_results(search_dirs: list[Path]) -> pd.DataFrame:
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
                    row[key] = val

            if "approach" not in row:
                row["approach"] = cfg["approach"]["_target_"].rsplit(".", 1)[-1]
            if "environment" not in row:
                row["environment"] = cfg["environment"]["_target_"].rsplit(".", 1)[-1]
            if "seed" not in row:
                row["seed"] = cfg["seed"]

            row.update({k: v for k, v in results.items() if k != "per_episode"})
            rows.append(row)

    return pd.DataFrame(rows)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="Hydra output directories to scan for results.json files.",
    )
    args = parser.parse_args()
    dataframe = _collect_results(args.dirs)
    if dataframe.empty:
        print("No results found.")
        return

    numeric_cols = dataframe.select_dtypes(include="number").columns.tolist()
    if "seed" in numeric_cols:
        numeric_cols.remove("seed")
    group_cols = [c for c in dataframe.columns if c not in numeric_cols and c != "seed"]

    seed_info = (
        dataframe.groupby(group_cols, sort=False)["seed"]
        .agg(seeds=lambda s: sorted(s.astype(str)))
        .reset_index()
    )
    seed_info["seeds"] = seed_info["seeds"].apply(",".join)
    seed_info["n_seeds"] = seed_info["seeds"].str.count(",") + 1

    averaged = (
        dataframe.drop(columns=["seed"])
        .groupby(group_cols, sort=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    averaged = averaged.merge(seed_info, on=group_cols)

    col_order = group_cols + ["n_seeds", "seeds"] + numeric_cols
    averaged = averaged[[c for c in col_order if c in averaged.columns]]

    sort_cols = ["environment"] + [c for c in group_cols if c != "environment"]
    averaged = averaged.sort_values(sort_cols).reset_index(drop=True)

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 200
    ):
        print(averaged.to_string(index=False))


if __name__ == "__main__":
    _main()
