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
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from experiments.tracker.constraints import (
    ExperimentConfig,
    Scalar,
    is_valid_experiment,
)
from experiments.tracker.schema import ALL_COLUMNS
from robocode.experiment_protocol import (
    EXPERIMENT_ID_DIGEST_LENGTH,
    EXPERIMENT_ID_SEPARATOR,
    validate_eval_seed,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONF_DIR = _REPO_ROOT / "experiments" / "conf"
_PREFERRED_DIMENSIONS = (
    "environment",
    "approach",
    "primitive_level",
    "approach.blackbox",
    "approach.blackbox_runtime",
    "approach/backend",
    "approach/completion",
    "eval_timeout",
)
_TOP_LEVEL_CHOICES = ("environment", "approach", "primitive_level")
_NESTED_APPROACH_CHOICES = ("approach/backend", "approach/completion")
_PROTOCOL_CONFIG_KEYS = ("experiment_id", "replicate_seed", "eval_seed")


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
        if key in _PROTOCOL_CONFIG_KEYS:
            raise ValueError(f"{path}: matrix cannot override protocol field {key!r}")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{path}: matrix.{key} must be a non-empty list")
        matrix[key] = [
            _require_scalar(choice, f"{path}: matrix.{key}") for choice in choices
        ]

    raw_seeds = raw.get("replicate_seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError(f"{path}: replicate_seeds must be a non-empty list")
    if not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in raw_seeds
    ):
        raise ValueError(f"{path}: every replicate seed must be an integer")
    replicate_seeds = tuple(raw_seeds)
    if len(replicate_seeds) != len(set(replicate_seeds)):
        raise ValueError(f"{path}: replicate_seeds must be unique")
    return name, matrix, replicate_seeds


def expand_matrix(
    campaign: str,
    matrix: Mapping[str, Sequence[Scalar]],
    replicate_seeds: tuple[int, ...],
) -> list[ExperimentConfig]:
    """Compute a campaign's requested Cartesian product in file order."""
    keys = tuple(matrix)
    return [
        ExperimentConfig(
            campaign,
            dict(zip(keys, choices, strict=True)),
            replicate_seeds,
        )
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


def _compose_with_hydra(
    config: ExperimentConfig, extra_overrides: Sequence[str] = ()
) -> DictConfig:
    """Compose one condition and report invalid overrides with useful context."""
    overrides = [*hydra_overrides(config), *extra_overrides]
    try:
        with initialize_config_dir(config_dir=str(_CONF_DIR), version_base=None):
            return compose(
                config_name="config",
                overrides=overrides,
                return_hydra_config=True,
            )
    except Exception as error:
        condition = ", ".join(overrides)
        raise ValueError(f"Hydra could not compose {condition}: {error}") from error


def _canonicalize_with_hydra(
    config: ExperimentConfig,
) -> tuple[ExperimentConfig, DictConfig]:
    """Materialize tracked Hydra defaults and retain their composed config."""
    cfg = _compose_with_hydra(config)
    choices = cfg.hydra.runtime.choices
    values = dict(config.values)
    for key in _TOP_LEVEL_CHOICES:
        choice = choices.get(key)
        if choice is not None:
            values[key] = str(choice)
    for key in _NESTED_APPROACH_CHOICES:
        choice = choices.get(key)
        if choice is None:
            values.pop(key, None)
        else:
            values[key] = str(choice)

    if "blackbox" in cfg.approach:
        blackbox = cfg.approach.blackbox
        if not isinstance(blackbox, bool):
            raise ValueError(
                f"Hydra must compose approach.blackbox to a boolean, got {blackbox!r}"
            )
        values["approach.blackbox"] = blackbox
    else:
        values.pop("approach.blackbox", None)

    if "blackbox_runtime" in cfg.approach and (
        values.get("approach.blackbox") is True
        or "approach.blackbox_runtime" in config.values
    ):
        runtime = cfg.approach.blackbox_runtime
        if runtime not in ("legacy", "strict"):
            raise ValueError(
                "Hydra must compose approach.blackbox_runtime to 'legacy' or "
                f"'strict', got {runtime!r}"
            )
        values["approach.blackbox_runtime"] = str(runtime)
    else:
        values.pop("approach.blackbox_runtime", None)

    eval_timeout = cfg.get("eval_timeout")
    if (
        not isinstance(eval_timeout, (int, float))
        or isinstance(eval_timeout, bool)
        or eval_timeout <= 0
    ):
        raise ValueError(
            "Hydra must compose eval_timeout to a positive number of seconds, "
            f"got {eval_timeout!r}"
        )
    normalized_timeout: int | float = (
        int(eval_timeout) if float(eval_timeout).is_integer() else float(eval_timeout)
    )
    values["eval_timeout"] = normalized_timeout
    cfg.eval_timeout = normalized_timeout
    return ExperimentConfig(config.campaign, values, config.replicate_seeds), cfg


def canonicalize_config(config: ExperimentConfig) -> ExperimentConfig:
    """Materialize tracked Hydra defaults and normalize their parsed values."""
    return _canonicalize_with_hydra(config)[0]


def validate_with_hydra(config: ExperimentConfig, eval_seed: int) -> None:
    """Compose the exact generated run, including its fixed protocol values."""
    eval_seed = validate_eval_seed(eval_seed)
    canonical, composed = _canonicalize_with_hydra(config)
    condition_id = _experiment_id(canonical, eval_seed, composed)
    _validate_canonical_run(canonical, condition_id, eval_seed)


def _validate_canonical_run(
    config: ExperimentConfig, condition_id: str, eval_seed: int
) -> None:
    """Compose a canonical condition with its generated protocol overrides."""
    _compose_with_hydra(
        config,
        (
            f"experiment_id={condition_id}",
            f"replicate_seed={config.replicate_seeds[0]}",
            f"eval_seed={eval_seed}",
        ),
    )


def _slug(value: Scalar) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return text or "unset"


def _experiment_id(
    config: ExperimentConfig, eval_seed: int, composed_cfg: DictConfig | None = None
) -> str:
    """Build an ID from one already-canonicalized executable run protocol."""
    labels: list[str] = []
    for key in _PREFERRED_DIMENSIONS:
        if key not in config.values:
            continue
        value = config.values[key]
        if key == "approach.blackbox":
            labels.append("blackbox" if value is True else "whitebox")
        elif key == "approach.blackbox_runtime":
            labels.append(f"bb_{_slug(value)}")
        elif key == "eval_timeout":
            labels.append(f"timeout_{_slug(value)}s")
        else:
            labels.append(_slug(value))

    cfg = composed_cfg if composed_cfg is not None else _compose_with_hydra(config)
    composed = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(composed, dict)
    composed.pop("hydra", None)
    for key in _PROTOCOL_CONFIG_KEYS:
        composed.pop(key, None)
    payload = {
        "config": composed,
        "replicate_seeds": sorted(config.replicate_seeds),
        "eval_seed": eval_seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
        :EXPERIMENT_ID_DIGEST_LENGTH
    ]
    return EXPERIMENT_ID_SEPARATOR.join([*(labels or ["condition"]), digest])


def experiment_id(config: ExperimentConfig, eval_seed: int) -> str:
    """Build a shared ID from Hydra's canonical config and the run protocol."""
    eval_seed = validate_eval_seed(eval_seed)
    canonical, composed = _canonicalize_with_hydra(config)
    return _experiment_id(canonical, eval_seed, composed)


def hydra_command(config: ExperimentConfig, condition_id: str, eval_seed: int) -> str:
    """Create one multirun command with fixed evaluation and swept replicates."""
    overrides = [
        *hydra_overrides(config),
        f"experiment_id={condition_id}",
        f"replicate_seed={','.join(map(str, config.replicate_seeds))}",
        f"eval_seed={eval_seed}",
        (f"hydra.sweep.dir=multirun/{condition_id}/${{now:%Y-%m-%d_%H-%M-%S}}"),
        "hydra.sweep.subdir=replicate_${replicate_seed}",
    ]
    quoted = " ".join(shlex.quote(override) for override in overrides)
    return f"python experiments/run_experiment.py -m {quoted}"


def _tracker_row(
    config: ExperimentConfig, condition_id: str, eval_seed: int
) -> dict[str, str]:
    """Convert one canonical condition to the shared tracker schema."""
    blackbox = config.values.get("approach.blackbox")
    access = ""
    if isinstance(blackbox, bool):
        if blackbox and config.values.get("approach.blackbox_runtime") == "strict":
            access = "strict-blackbox"
        else:
            access = "blackbox" if blackbox else "whitebox"
    row = {column: "" for column in ALL_COLUMNS}
    row.update(
        {
            "Experiment ID": condition_id,
            "Campaign": config.campaign,
            "Environment": str(config.values.get("environment", "")),
            "Method": str(config.values.get("approach", "")),
            "Primitive Level": str(config.values.get("primitive_level", "")),
            "Access": access,
            "Model / Backend": str(
                config.values.get(
                    "approach/backend",
                    config.values.get("approach/completion", ""),
                )
            ),
            "Replicate Seeds": json.dumps(config.replicate_seeds),
            "Evaluation Seed": str(eval_seed),
            "Command": hydra_command(config, condition_id, eval_seed),
            "Active": "TRUE",
            "Status": "Todo",
            "Progress": f"0/{len(config.replicate_seeds)}",
        }
    )
    return row


def tracker_row(config: ExperimentConfig, eval_seed: int) -> dict[str, str]:
    """Convert a valid condition to the shared tracker schema."""
    eval_seed = validate_eval_seed(eval_seed)
    canonical, composed = _canonicalize_with_hydra(config)
    condition_id = _experiment_id(canonical, eval_seed, composed)
    return _tracker_row(canonical, condition_id, eval_seed)


def generate_rows(
    campaign_paths: Iterable[Path],
    eval_seed: int,
    validate_hydra: bool = True,
) -> tuple[list[dict[str, str]], list[tuple[ExperimentConfig, str, str]]]:
    """Expand campaigns, exclude constraints, and return tracker rows."""
    eval_seed = validate_eval_seed(eval_seed)
    rows: list[dict[str, str]] = []
    excluded: list[tuple[ExperimentConfig, str, str]] = []
    seen_ids: dict[str, str] = {}
    for path in campaign_paths:
        campaign, matrix, replicate_seeds = load_campaign(path)
        for requested_config in expand_matrix(campaign, matrix, replicate_seeds):
            config, composed = _canonicalize_with_hydra(requested_config)
            condition_id = _experiment_id(config, eval_seed, composed)
            valid, reason = is_valid_experiment(config)
            if not valid:
                assert reason is not None
                excluded.append((config, reason, condition_id))
                continue
            if validate_hydra:
                _validate_canonical_run(config, condition_id, eval_seed)
            row = _tracker_row(config, condition_id, eval_seed)
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
    excluded: Sequence[tuple[ExperimentConfig, str, str]],
    list_conditions: bool,
) -> None:
    print(f"{len(rows)} valid experiment conditions")
    print(f"{len(excluded)} invalid combinations excluded")
    if list_conditions:
        for row in rows:
            print(f"+ {row['Experiment ID']}")
        for _, reason, condition_id in excluded:
            print(f"SKIP {condition_id}")
            print(f"     {reason}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", type=Path, nargs="+")
    parser.add_argument(
        "--eval-seed",
        type=int,
        required=True,
        help="Private fixed evaluation-suite seed (do not add it to campaign YAML)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-hydra-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the campaign generator CLI."""
    args = _parse_args()
    rows, excluded = generate_rows(
        args.campaigns,
        eval_seed=args.eval_seed,
        validate_hydra=not args.no_hydra_validation,
    )
    _summary(rows, excluded, list_conditions=args.dry_run)
    if not args.dry_run:
        output = args.output or _default_output(args.campaigns)
        write_csv(rows, output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
