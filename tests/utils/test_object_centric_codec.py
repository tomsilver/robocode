"""Tests for the ObjectCentricState blackbox codec and the object-centric wire loop."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
from typing import Any

import numpy as np
import pytest
from kinder.envs.kinematic2d.object_types import RectangleType
from omegaconf import OmegaConf

from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.primitives import blackbox_primitive_manifest
from robocode.utils import env_server
from robocode.utils.env_server import env_server_running, write_env_spaces
from robocode.utils.object_centric_codec import (
    decode_object_centric_state,
    encode_object_centric_state,
    serialize_object_centric_space,
)

_OBSTRUCTION2D_PATH = "kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv"


def _env() -> VariableObjectCountEnv:
    return VariableObjectCountEnv(
        constant_object_env_path=_OBSTRUCTION2D_PATH,
        count_kwarg="num_obstructions",
        count_object_prefix="obstruction",
        design_counts=[0, 1, 2],
        eval_counts=[0, 1, 2, 3, 4],
        bilevel_env_name="obstruction2d",
    )


def test_codec_round_trip_is_json_safe_and_faithful() -> None:
    """Encode -> JSON -> decode reproduces the state's objects, features, and types."""
    env = _env()
    for count in (1, 4):  # a design count and a held-out one
        state, _ = env.reset(seed=count, options={"object_count": count})
        payload = json.loads(json.dumps(encode_object_centric_state(state)))
        back = decode_object_centric_state(payload)
        assert back.allclose(state)
        assert set(back.get_object_names()) == set(state.get_object_names())
        # get(obj, feature) works -> type_features preserved.
        for name in state.get_object_names():
            obj_a = state.get_object_from_name(name)
            obj_b = back.get_object_from_name(name)
            for feat in state.type_features[obj_a.type]:
                assert np.isclose(state.get(obj_a, feat), back.get(obj_b, feat))


def test_codec_preserves_type_ancestors() -> None:
    """is_instance over ancestors survives the round trip (subtype rectangles match)."""
    env = _env()
    state, _ = env.reset(seed=0, options={"object_count": 2})
    back = decode_object_centric_state(
        json.loads(json.dumps(encode_object_centric_state(state)))
    )
    real = {o.name for o in state.get_objects(RectangleType)}
    back_rect = next(
        o.type
        for o in (back.get_object_from_name(n) for n in back.get_object_names())
        if o.type.name == "rectangle"
    )
    assert {o.name for o in back.get_objects(back_rect)} == real
    # target_block/target_surface are subtypes of rectangle, so they are included.
    assert {"target_block", "target_surface"} <= real


def test_serialize_object_centric_space_has_types_and_parents() -> None:
    """serialize_space emits an ObjectCentric schema with a real type hierarchy."""
    env = _env()
    schema = serialize_object_centric_space(env.type_features)
    assert schema["type"] == "ObjectCentric"
    assert any(t["parent"] is not None for t in schema["types"])  # a real hierarchy
    # env_server.serialize_space dispatches to the ObjectCentric branch.
    assert env_server.serialize_space(env.observation_space)["type"] == "ObjectCentric"


def _load_env_client(metadata_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        "env_client_undertest", "src/robocode/utils/env_client.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.make_env(metadata_path)


def test_blackbox_object_centric_loop() -> None:
    """A full client<->server loop carries object-centric states over the wire."""
    cfg = OmegaConf.load("experiments/conf/environment/obstruction2d_generalized.yaml")
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    kwargs: dict[str, Any] = {
        str(k): v for k, v in container.items() if k != "_target_"
    }
    host_env = VariableObjectCountEnv(**kwargs)
    env_cfg_json = json.dumps(OmegaConf.to_container(cfg, resolve=True))

    with tempfile.TemporaryDirectory() as tmp:
        sandbox = pathlib.Path(tmp) / "sandbox"
        sandbox.mkdir()
        with env_server_running(env_cfg_json, sandbox) as (port, token):
            write_env_spaces(
                sandbox,
                container_backend="local",
                port=port,
                token=token,
                observation_space=host_env.observation_space,
                action_space=host_env.action_space,
                max_steps=200,
                primitives_manifest=blackbox_primitive_manifest(
                    ["check_action_collision"]
                ),
            )
            client_mod, env = _load_env_client(sandbox / "env_spaces.json")
            # pylint: disable-next=protected-access
            local_state_cls = client_mod._ObjectCentricState

            # Held-out count 4 arrives as a local object-centric state.
            obs, info = env.reset(seed=3, options={"object_count": 4})
            assert isinstance(obs, local_state_cls)
            assert info["object_count"] == 4
            n = sum(1 for nm in obs.get_object_names() if nm.startswith("obstruction"))
            assert n == 4
            rect = env.observation_space.get_type("rectangle")
            assert len(obs.get_objects(rect)) == 6  # 4 obstructions + block + surface

            obs2, _reward, _term, _trunc, _info = env.step(env.action_space.sample())
            assert isinstance(obs2, local_state_cls)

            snapshot = env.get_state()
            assert isinstance(snapshot, local_state_cls)
            env.step(env.action_space.sample())
            env.set_state(snapshot)
            assert snapshot.allclose(env.get_state())

            collision = env.make_primitives()["check_action_collision"](
                env.get_state(), env.action_space.sample()
            )
            assert isinstance(collision, bool)
            env.close()


def test_blackbox_bilevel_models_rejected() -> None:
    """bilevel_models has no blackbox host proxy; requesting it fails loudly."""
    with pytest.raises(ValueError, match="bilevel_models"):
        blackbox_primitive_manifest(["bilevel_models"])
