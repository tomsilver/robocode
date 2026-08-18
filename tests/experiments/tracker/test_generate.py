"""Tests for Hydra-backed campaign expansion and canonical IDs."""

import shlex
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

from experiments.tracker import generate
from robocode.experiment_protocol import EXPERIMENT_ID_PATTERN

_REPO_ROOT = Path(__file__).resolve().parents[3]

_TEST_EVAL_SEED = 918273645
_SAMPLE_CAMPAIGN_YAML = """name: test_campaign
matrix:
  environment: [motion2d_easy]
  approach: [agentic]
  primitive_level: [none, low_level, bilevel]
  approach.blackbox: [false, true]
  approach/backend: [claude_opus5]
replicate_seeds: [42, 24]
"""


def _sample_campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(_SAMPLE_CAMPAIGN_YAML, encoding="utf-8")
    return campaign


def _config(**updates: Any) -> Any:
    values = {
        "environment": "motion2d_easy",
        "approach": "agentic",
        "primitive_level": "none",
        **updates,
    }
    return generate.ExperimentConfig("campaign_a", values, (42, 24))


def test_sample_campaign_expansion_and_constraint(tmp_path: Path) -> None:
    """The sample campaign yields five rows and excludes one invalid matrix cell."""
    rows, excluded = generate.generate_rows(
        [_sample_campaign(tmp_path)], eval_seed=_TEST_EVAL_SEED
    )
    assert len(rows) == 5
    assert len(excluded) == 1
    assert excluded[0][1] == "bilevel_models unavailable under blackbox"
    assert all(row["Replicate Seeds"] == "[42, 24]" for row in rows)
    assert all(row["Evaluation Seed"] == str(_TEST_EVAL_SEED) for row in rows)
    assert all("replicate_seed=42,24" in row["Command"] for row in rows)
    assert all(f"eval_seed={_TEST_EVAL_SEED}" in row["Command"] for row in rows)
    assert all("eval_timeout=60" in row["Command"] for row in rows)


@pytest.mark.parametrize(
    "protocol_key", ["replicate_seed", "eval_seed", "experiment_id"]
)
def test_campaign_matrix_rejects_protocol_fields(
    protocol_key: str, tmp_path: Path
) -> None:
    """Campaigns cannot collide with generator-owned protocol overrides."""
    campaign = tmp_path / "bad-protocol.yaml"
    campaign.write_text(
        "name: bad\nmatrix:\n"
        "  environment: [motion2d_easy]\n"
        f"  {protocol_key}: [1]\n"
        "replicate_seeds: [42]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot override protocol field"):
        generate.load_campaign(campaign)


@pytest.mark.parametrize("eval_seed", [None, True, -1, 42.5, "42"])
def test_generator_rejects_non_integer_eval_seed(
    eval_seed: Any, tmp_path: Path
) -> None:
    """Programmatic callers cannot silently coerce the private protocol seed."""
    with pytest.raises(ValueError, match="eval_seed"):
        generate.generate_rows(
            [_sample_campaign(tmp_path)], eval_seed=eval_seed, validate_hydra=False
        )


def test_generated_command_propagates_id_to_hydra_and_output_path() -> None:
    """The Sheet ID is also runtime metadata and the artifact parent directory."""
    row = generate.tracker_row(_config(), eval_seed=_TEST_EVAL_SEED)
    condition_id = row["Experiment ID"]
    command = shlex.split(row["Command"])
    assert EXPERIMENT_ID_PATTERN.fullmatch(condition_id)
    assert f"experiment_id={condition_id}" in command
    assert "replicate_seed=42,24" in command
    assert f"eval_seed={_TEST_EVAL_SEED}" in command
    assert (
        f"hydra.sweep.dir=multirun/{condition_id}/${{now:%Y-%m-%d_%H-%M-%S}}"
    ) in command
    assert "hydra.sweep.subdir=replicate_${replicate_seed}" in command


def test_experiment_id_is_independent_of_mapping_and_replicate_order() -> None:
    """Mapping and replicate order do not change the canonical run ID."""
    first = _config(primitive_level="low_level")
    second = generate.ExperimentConfig(
        "campaign_b", dict(reversed(tuple(first.values.items()))), (24, 42)
    )
    assert generate.experiment_id(first, _TEST_EVAL_SEED) == generate.experiment_id(
        second, _TEST_EVAL_SEED
    )


def test_experiment_id_changes_with_seed_protocol() -> None:
    """A changed replicate set or evaluation suite appends a distinct run."""
    config = _config()
    changed_replicates = generate.ExperimentConfig(
        config.campaign, config.values, (42, 24, 424)
    )
    ids = {
        generate.experiment_id(config, _TEST_EVAL_SEED),
        generate.experiment_id(config, _TEST_EVAL_SEED + 1),
        generate.experiment_id(changed_replicates, _TEST_EVAL_SEED),
    }
    assert len(ids) == 3


def test_hydra_defaults_are_canonicalized_for_ids_metadata_and_commands() -> None:
    """Omitted tracked defaults are identical to the same explicit condition."""
    implicit = _config()
    explicit = _config(
        **{
            "approach.blackbox": False,
            "approach/backend": "claude_opus5",
            "eval_timeout": 60,
        }
    )
    implicit_row = generate.tracker_row(implicit, eval_seed=_TEST_EVAL_SEED)
    explicit_row = generate.tracker_row(explicit, eval_seed=_TEST_EVAL_SEED)
    assert implicit_row["Experiment ID"] == explicit_row["Experiment ID"]
    assert implicit_row["Access"] == "whitebox"
    assert implicit_row["Model / Backend"] == "claude_opus5"
    command = shlex.split(implicit_row["Command"])
    assert "approach.blackbox=false" in command
    assert "approach/backend=claude_opus5" in command
    assert "eval_timeout=60" in command


def test_equivalent_integer_and_float_timeouts_share_an_id() -> None:
    """YAML numeric spelling does not create duplicate executable conditions."""
    integer = _config(eval_timeout=60)
    floating = _config(eval_timeout=60.0)
    assert generate.experiment_id(integer, _TEST_EVAL_SEED) == generate.experiment_id(
        floating, _TEST_EVAL_SEED
    )


def test_timeout_change_produces_a_distinct_experiment_id() -> None:
    """Evaluation time is executable condition data, not an ambient default."""
    assert generate.experiment_id(_config(), _TEST_EVAL_SEED) != (
        generate.experiment_id(_config(eval_timeout=30), _TEST_EVAL_SEED)
    )


def test_quoted_blackbox_true_is_normalized_before_constraints(tmp_path: Path) -> None:
    """YAML strings that Hydra parses as booleans cannot bypass validity rules."""
    campaign = tmp_path / "quoted-blackbox.yaml"
    campaign.write_text(
        """name: quoted_blackbox
matrix:
  environment: [motion2d_easy]
  approach: [agentic]
  primitive_level: [bilevel]
  approach.blackbox: ["true"]
replicate_seeds: [42]
""",
        encoding="utf-8",
    )
    rows, excluded = generate.generate_rows([campaign], eval_seed=_TEST_EVAL_SEED)
    assert not rows
    assert excluded[0][0].values["approach.blackbox"] is True
    assert excluded[0][1] == "bilevel_models unavailable under blackbox"


def test_missing_hydra_choice_fails_generation(tmp_path: Path) -> None:
    """Campaign typos fail at Hydra composition instead of reaching the Sheet."""
    campaign = tmp_path / "bad.yaml"
    campaign.write_text(
        """name: typo
matrix:
  environment: [not_an_environment]
  approach: [agentic]
  primitive_level: [none]
replicate_seeds: [42]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Hydra could not compose"):
        generate.generate_rows([campaign], eval_seed=_TEST_EVAL_SEED)


@pytest.mark.parametrize(
    ("primitive_level", "expected"),
    [
        ("none", []),
        ("low_level", ["check_action_collision", "BiRRT"]),
        ("bilevel", ["bilevel_models"]),
    ],
)
def test_primitive_level_composes_to_implementation_list(
    primitive_level: str, expected: list[str]
) -> None:
    """The semantic Hydra group remains the single source for primitive lists."""
    config = _config(primitive_level=primitive_level)
    with initialize_config_dir(
        config_dir=str(_REPO_ROOT / "experiments" / "conf"), version_base=None
    ):
        cfg = compose(config_name="config", overrides=generate.hydra_overrides(config))
    assert cfg.primitive_level == primitive_level
    assert list(cfg.primitives) == expected


def test_generation_composes_each_valid_condition_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonicalization and exact-run validation each compose once."""
    campaign = tmp_path / "single.yaml"
    campaign.write_text(
        """name: single
matrix:
  environment: [motion2d_easy]
  approach: [agentic]
  primitive_level: [none]
replicate_seeds: [42]
""",
        encoding="utf-8",
    )
    original = generate._compose_with_hydra  # pylint: disable=protected-access
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(generate, "_compose_with_hydra", counted)
    rows, excluded = generate.generate_rows([campaign], eval_seed=_TEST_EVAL_SEED)
    assert len(rows) == 1
    assert not excluded
    assert calls == 2
