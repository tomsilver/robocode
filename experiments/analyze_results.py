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
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 200
    ):
        print(dataframe.to_string(index=False))


if __name__ == "__main__":
    _main()
