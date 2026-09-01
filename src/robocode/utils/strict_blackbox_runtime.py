"""Generic runtime for evaluating a strict-blackbox generated policy.

This file is copied into the strict Docker image.  It deliberately imports only
the standard library and NumPy: no RoboCode, KinDER, geometry, simulator, or
robotics package is available to the generated program at evaluation time.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_NDARRAY_TAG = "__ndarray__"
_OCS_TAG = "__ocs__"


class TypeInfo:
    """Minimal object-type value used by object-centric observations."""

    def __init__(self, name: str, parent: "TypeInfo | None" = None) -> None:
        self.name = name
        self.parent = parent

    def get_ancestors(self) -> set["TypeInfo"]:
        """Return this type and every transitive parent."""
        ancestors: set[TypeInfo] = set()
        current: TypeInfo | None = self
        while current is not None:
            ancestors.add(current)
            current = current.parent
        return ancestors

    def __eq__(self, other: Any) -> bool:
        """Compare types by public name."""
        return isinstance(other, TypeInfo) and self.name == other.name

    def __hash__(self) -> int:
        """Hash consistently with name-based equality."""
        return hash(self.name)


class ObjectInfo:
    """Minimal object value used by object-centric observations."""

    def __init__(self, name: str, typ: TypeInfo) -> None:
        self.name = name
        self.type = typ

    def is_instance(self, typ: TypeInfo) -> bool:
        """Return whether this object has *typ* in its ancestor chain."""
        return typ in self.type.get_ancestors()

    def __eq__(self, other: Any) -> bool:
        """Compare objects by public name."""
        return isinstance(other, ObjectInfo) and self.name == other.name

    def __hash__(self) -> int:
        """Hash consistently with name-based equality."""
        return hash(self.name)


class ObjectCentricStateInfo:
    """Dependency-free mirror of the public ObjectCentricState read API."""

    def __init__(
        self,
        data: dict[ObjectInfo, np.ndarray],
        type_features: dict[TypeInfo, list[str]],
    ) -> None:
        self.data = data
        self.type_features = type_features

    def __iter__(self) -> Any:
        """Iterate through objects in stable name order."""
        return iter(sorted(self.data, key=lambda obj: obj.name))

    def get_object_names(self) -> set[str]:
        """Return the names present in this state."""
        return {obj.name for obj in self.data}

    def get_object_from_name(self, name: str) -> ObjectInfo:
        """Look up an object by name."""
        for obj in self.data:
            if obj.name == name:
                return obj
        raise ValueError(f"No object named {name!r}")

    def get_objects(self, typ: TypeInfo) -> list[ObjectInfo]:
        """Return objects that are instances of *typ*."""
        return sorted(
            (obj for obj in self.data if obj.is_instance(typ)),
            key=lambda obj: obj.name,
        )

    def get(self, obj: ObjectInfo, feature: str) -> float:
        """Read one named object feature."""
        return float(self.data[obj][self.type_features[obj.type].index(feature)])

    def set(self, obj: ObjectInfo, feature: str, value: float) -> None:
        """Set one named object feature."""
        self.data[obj][self.type_features[obj.type].index(feature)] = value

    def vec(self, objects: Any) -> np.ndarray:
        """Concatenate the features of *objects*."""
        return (
            np.concatenate([self.data[obj] for obj in objects])
            if objects
            else np.array([])
        )

    def copy(self) -> "ObjectCentricStateInfo":
        """Return a deep copy of this state."""
        return ObjectCentricStateInfo(
            {obj: values.copy() for obj, values in self.data.items()},
            {typ: list(features) for typ, features in self.type_features.items()},
        )


def _build_types(
    payload: list[dict[str, Any]],
) -> tuple[dict[str, TypeInfo], dict[TypeInfo, list[str]]]:
    by_name: dict[str, TypeInfo] = {}
    features: dict[TypeInfo, list[str]] = {}
    for entry in payload:
        parent = by_name[entry["parent"]] if entry["parent"] is not None else None
        typ = TypeInfo(entry["name"], parent)
        by_name[typ.name] = typ
        features[typ] = list(entry["features"])
    return by_name, features


def _decode_ocs(payload: dict[str, Any]) -> ObjectCentricStateInfo:
    by_name, type_features = _build_types(payload["types"])
    data: dict[ObjectInfo, np.ndarray] = {}
    for entry in payload["objects"]:
        obj = ObjectInfo(entry["name"], by_name[entry["type"]])
        data[obj] = np.asarray(entry["features"], dtype=np.float32)
    return ObjectCentricStateInfo(data, type_features)


def decode(obj: Any) -> Any:
    """Decode the host's tagged JSON values."""
    if isinstance(obj, dict):
        if _NDARRAY_TAG in obj:
            return np.asarray(obj[_NDARRAY_TAG], dtype=np.dtype(obj["dtype"]))
        if _OCS_TAG in obj:
            return _decode_ocs(obj[_OCS_TAG])
        return {key: decode(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [decode(value) for value in obj]
    return obj


def encode(obj: Any) -> Any:
    """Encode policy results as tagged JSON values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {_NDARRAY_TAG: obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(key): encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(value) for value in obj]
    raise TypeError(f"Strict policy returned unsupported type {type(obj).__name__}")


class BoxSpaceInfo:
    """Generic, dependency-free representation of a gymnasium Box."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.shape = tuple(spec["shape"])
        self.dtype = np.dtype(spec["dtype"])
        self.low = np.asarray(spec["low"], dtype=self.dtype)
        self.high = np.asarray(spec["high"], dtype=self.dtype)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample uniformly within the box bounds."""
        generator = rng if rng is not None else np.random.default_rng()
        return generator.uniform(self.low, self.high).astype(self.dtype)


class ObjectCentricSpaceInfo:
    """Generic representation of ObjectCentricStateSpace metadata."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self._by_name, self._type_features = _build_types(spec["types"])

    @property
    def types(self) -> set[TypeInfo]:
        """Return the domain's object types."""
        return set(self._type_features)

    @property
    def type_features(self) -> dict[TypeInfo, list[str]]:
        """Return a copy of the type-to-feature mapping."""
        return {typ: list(features) for typ, features in self._type_features.items()}

    def get_type(self, name: str) -> TypeInfo:
        """Look up an object type by name."""
        return self._by_name[name]


def make_space(spec: dict[str, Any]) -> Any:
    """Build a generic space object from public metadata."""
    if spec["type"] == "Box":
        return BoxSpaceInfo(spec)
    if spec["type"] == "ObjectCentric":
        return ObjectCentricSpaceInfo(spec)
    raise TypeError(f"Unsupported strict-blackbox space type {spec['type']!r}")


def _load_policy(policy_path: Path, metadata: dict[str, Any]) -> Any:
    policy_dir = str(policy_path.parent.resolve())
    sys.path.insert(0, policy_dir)
    try:
        namespace: dict[str, Any] = {"__file__": str(policy_path)}
        source = policy_path.read_text(encoding="utf-8")
        exec(  # pylint: disable=exec-used
            compile(source, str(policy_path), "exec"), namespace
        )
        return namespace["GeneratedApproach"](
            make_space(metadata["action_space"]),
            make_space(metadata["observation_space"]),
            primitives={},
        )
    finally:
        sys.path.remove(policy_dir)


def main() -> None:
    """Serve the policy lifecycle over JSON lines on stdin/stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/policy/approach.py")
    args = parser.parse_args()
    policy_path = Path(args.policy)
    policy: Any = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            should_close = False
            # Generated code may print. Keep stdout exclusively for the protocol.
            with contextlib.redirect_stdout(sys.stderr):
                command = request["cmd"]
                if command == "init":
                    policy = _load_policy(policy_path, request)
                    result = None
                elif command == "reset":
                    policy.reset(decode(request["state"]), decode(request["info"]))
                    result = None
                elif command == "get_action":
                    result = policy.get_action(decode(request["state"]))
                elif command == "update":
                    if hasattr(policy, "update"):
                        policy.update(
                            decode(request["state"]),
                            float(request["reward"]),
                            bool(request["done"]),
                            decode(request["info"]),
                        )
                    result = None
                elif command == "close":
                    result = None
                    should_close = True
                else:
                    raise ValueError(f"Unknown strict-policy command {command!r}")
            print(json.dumps({"ok": True, "result": encode(result)}), flush=True)
            if should_close:
                return
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            # Policy failures, including timeout signals, must cross the boundary.
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
