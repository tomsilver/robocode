"""Expand small campaign matrices into tracker rows and Hydra commands."""

import argparse
import csv
import hashlib
import itertools
import json
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from constraints import ExperimentConfig, Scalar, is_valid_experiment
from hydra import compose, initialize_config_dir
from schema import ALL_COLUMNS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONF_DIR = _REPO_ROOT / "experiments" / "conf"
_PREFERRED_DIMENSIONS = (
    "environment",
    "approach",
    "primitive_level",
    "approach.blackbox",
    "approach/backend",
)


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{description} keys must be strings")
    return value


def _require_scalar(value: Any, description: str) -> Scalar:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"{description} must be a scalar")


def load_campaign(path: Path) -> tuple[str, dict[str, list[Scalar]], tuple[int, ...]]:
    """Load and validate the small campaign-file schema."""
    raw = _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}: name must be a non-empty string")

    raw_matrix = _require_mapping(raw.get("matrix"), f"{path}: matrix")
    matrix: dict[str, list[Scalar]] = {}
    for key, choices in raw_matrix.items():
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{path}: matrix.{key} must be a non-empty list")
        matrix[key] = [
            _require_scalar(choice, f"{path}: matrix.{key}") for choice in choices
        ]

    raw_seeds = raw.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError(f"{path}: seeds must be a non-empty list")
    if not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in raw_seeds
    ):
        raise ValueError(f"{path}: every seed must be an integer")
    seeds = tuple(raw_seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{path}: seeds must be unique")
    return name, matrix, seeds


def expand_matrix(
    campaign: str, matrix: Mapping[str, Sequence[Scalar]], seeds: tuple[int, ...]
) -> list[ExperimentConfig]:
    """Compute a campaign's requested Cartesian product in file order."""
    keys = tuple(matrix)
    return [
        ExperimentConfig(campaign, dict(zip(keys, choices, strict=True)), seeds)
        for choices in itertools.product(*(matrix[key] for key in keys))
    ]


def _override_value(value: Scalar) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def _ordered_keys(values: Mapping[str, Scalar]) -> list[str]:
    preferred = [key for key in _PREFERRED_DIMENSIONS if key in values]
    return preferred + sorted(set(values) - set(preferred))


def hydra_overrides(config: ExperimentConfig) -> list[str]:
    """Translate a condition into Hydra override syntax."""
    return [
        f"{key}={_override_value(config.values[key])}"
        for key in _ordered_keys(config.values)
    ]


def validate_with_hydra(config: ExperimentConfig) -> None:
    """Compose a condition so missing config choices fail during generation."""
    try:
        with initialize_config_dir(config_dir=str(_CONF_DIR), version_base=None):
            compose(config_name="config", overrides=hydra_overrides(config))
    except Exception as error:
        condition = ", ".join(hydra_overrides(config))
        raise ValueError(f"Hydra could not compose {condition}: {error}") from error


def _slug(value: Scalar) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return text or "unset"


def experiment_id(config: ExperimentConfig) -> str:
    """Build a readable ID plus a hash of every canonical matrix dimension."""
    labels: list[str] = []
    for key in _PREFERRED_DIMENSIONS:
        if key not in config.values:
            continue
        value = config.values[key]
        if key == "approach.blackbox":
            labels.append("blackbox" if value is True else "whitebox")
        else:
            labels.append(_slug(value))
    canonical = json.dumps(config.values, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return "__".join([*(labels or ["condition"]), digest])


def hydra_command(config: ExperimentConfig) -> str:
    """Create one copyable Hydra multirun command for all required seeds."""
    overrides = [*hydra_overrides(config), f"seed={','.join(map(str, config.seeds))}"]
    quoted = " ".join(shlex.quote(override) for override in overrides)
    return f"python experiments/run_experiment.py -m {quoted}"


def tracker_row(config: ExperimentConfig) -> dict[str, str]:
    """Convert a valid condition to the shared tracker schema."""
    blackbox = config.values.get("approach.blackbox")
    access = ""
    if isinstance(blackbox, bool):
        access = "blackbox" if blackbox else "whitebox"
    row = {column: "" for column in ALL_COLUMNS}
    row.update(
        {
            "Experiment ID": experiment_id(config),
            "Campaign": config.campaign,
            "Environment": str(config.values.get("environment", "")),
            "Method": str(config.values.get("approach", "")),
            "Primitive Level": str(config.values.get("primitive_level", "")),
            "Access": access,
            "Model / Backend": str(config.values.get("approach/backend", "")),
            "Seeds": json.dumps(config.seeds),
            "Command": hydra_command(config),
            "Active": "TRUE",
            "Status": "Todo",
            "Progress": f"0/{len(config.seeds)}",
        }
    )
    return row


def generate_rows(
    campaign_paths: Iterable[Path], validate_hydra: bool = True
) -> tuple[list[dict[str, str]], list[tuple[ExperimentConfig, str]]]:
    """Expand campaigns, exclude constraints, and return tracker rows."""
    rows: list[dict[str, str]] = []
    excluded: list[tuple[ExperimentConfig, str]] = []
    seen_ids: dict[str, str] = {}
    for path in campaign_paths:
        campaign, matrix, seeds = load_campaign(path)
        for config in expand_matrix(campaign, matrix, seeds):
            valid, reason = is_valid_experiment(config)
            if not valid:
                assert reason is not None
                excluded.append((config, reason))
                continue
            if validate_hydra:
                validate_with_hydra(config)
            row = tracker_row(config)
            condition_id = row["Experiment ID"]
            if condition_id in seen_ids:
                raise ValueError(
                    f"Experiment ID {condition_id} occurs in both "
                    f"{seen_ids[condition_id]} and {campaign}"
                )
            seen_ids[condition_id] = campaign
            rows.append(row)
    return rows, excluded


def write_csv(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    """Write generated rows using the complete tracker column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _default_output(campaign_paths: Sequence[Path]) -> Path:
    stem = campaign_paths[0].stem if len(campaign_paths) == 1 else "campaigns"
    return _REPO_ROOT / "experiments" / "generated" / f"{stem}.csv"


def _summary(
    rows: Sequence[Mapping[str, str]],
    excluded: Sequence[tuple[ExperimentConfig, str]],
    list_conditions: bool,
) -> None:
    print(f"{len(rows)} valid experiment conditions")
    print(f"{len(excluded)} invalid combinations excluded")
    if list_conditions:
        for row in rows:
            print(f"+ {row['Experiment ID']}")
        for config, reason in excluded:
            print(f"SKIP {experiment_id(config)}")
            print(f"     {reason}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-hydra-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the campaign generator CLI."""
    args = _parse_args()
    rows, excluded = generate_rows(
        args.campaigns, validate_hydra=not args.no_hydra_validation
    )
    _summary(rows, excluded, list_conditions=args.dry_run)
    if not args.dry_run:
        output = args.output or _default_output(args.campaigns)
        write_csv(rows, output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
