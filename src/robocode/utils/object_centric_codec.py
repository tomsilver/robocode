"""JSON codec for ``ObjectCentricState`` over the blackbox env protocol.

A variable-count env hands the program an ``ObjectCentricState`` rather than a flat
vector, so the blackbox wire must carry that object -- the existing protocol only
knows ``Box`` vectors. This module encodes an ``ObjectCentricState`` to a
self-contained JSON payload and back to a real state on the host.

The payload embeds the type hierarchy (names + parents + feature lists) alongside the
per-object features, so it can be reconstructed on either side without any external
schema: the host rebuilds a real ``relational_structs.ObjectCentricState`` (this
module), and the sandbox client rebuilds a lightweight local mirror with the same API
(``env_client``). Types are ordered parents-first so ancestors exist before children
(needed to preserve ``is_instance``).
"""

from __future__ import annotations

from operator import attrgetter
from typing import Any

import numpy as np
from relational_structs import Object, ObjectCentricState, Type

OCS_TAG = "__ocs__"


def _ordered_types(type_features: dict[Type, list[str]]) -> list[Type]:
    """Return the types parents-first (topological by ancestor depth)."""

    def depth(t: Type) -> int:
        n, parent = 0, t.parent
        while parent is not None:
            n, parent = n + 1, parent.parent
        return n

    return sorted(type_features, key=lambda t: (depth(t), t.name))


def encode_object_centric_state(state: ObjectCentricState) -> dict[str, Any]:
    """Encode a state as a self-contained JSON-safe payload."""
    type_features = state.type_features
    types_payload = [
        {
            "name": t.name,
            "parent": t.parent.name if t.parent is not None else None,
            "features": list(type_features[t]),
        }
        for t in _ordered_types(type_features)
    ]
    objects_payload = []
    # Sort by name explicitly rather than relying on Object's rich comparison.
    for obj in sorted(state, key=attrgetter("name")):
        objects_payload.append(
            {
                "name": obj.name,
                "type": obj.type.name,
                "features": [float(v) for v in state.vec([obj])],
            }
        )
    return {
        "types": types_payload,
        "objects": objects_payload,
        "state_cls": type(state).__name__,
    }


def _rebuild_types(
    types_payload: list[dict[str, Any]],
) -> tuple[dict[str, Type], dict[Type, list[str]]]:
    """Rebuild ``{name: Type}`` and ``{Type: [features]}`` from the payload.

    The payload is parents-first, so each parent Type exists before its children.
    """
    type_by_name: dict[str, Type] = {}
    type_features: dict[Type, list[str]] = {}
    for entry in types_payload:
        parent_name = entry["parent"]
        parent = type_by_name[parent_name] if parent_name is not None else None
        typ = Type(entry["name"], parent)
        type_by_name[entry["name"]] = typ
        type_features[typ] = list(entry["features"])
    return type_by_name, type_features


def decode_object_centric_state(payload: dict[str, Any]) -> ObjectCentricState:
    """Rebuild a real ``ObjectCentricState`` from a codec payload (host side)."""
    type_by_name, type_features = _rebuild_types(payload["types"])
    data: dict[Object, np.ndarray] = {}
    for entry in payload["objects"]:
        obj = Object(entry["name"], type_by_name[entry["type"]])
        data[obj] = np.array(entry["features"], dtype=np.float32)
    return ObjectCentricState(data, type_features)


def serialize_object_centric_space(
    type_features: dict[Type, list[str]],
) -> dict[str, Any]:
    """Serialize an ``ObjectCentricStateSpace`` for the sandbox's env_spaces.json.

    Mirrors the encode payload's ``types`` schema (parents-first) so the client can
    build its local type mirror and expose ``observation_space.types``. The types are
    exactly the keys of *type_features*.
    """
    return {
        "type": "ObjectCentric",
        "types": [
            {
                "name": t.name,
                "parent": t.parent.name if t.parent is not None else None,
                "features": list(type_features[t]),
            }
            for t in _ordered_types(type_features)
        ],
    }
