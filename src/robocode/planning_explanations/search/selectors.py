"""Literal current-value selections; no expressions, calls or solver in mappings."""

# Exact built-in types are the restricted representation contract.
# pylint: disable=unidiomatic-typecheck

import ast
import json
from collections import deque
from collections.abc import Iterable
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4096)
def parse(selector: str) -> tuple[str, tuple[str | int, ...]]:
    """Accept a local followed only by literal string/integer subscripts."""
    try:
        node = ast.parse(selector, mode="eval").body
        keys = []
        while isinstance(node, ast.Subscript):
            key = ast.literal_eval(node.slice)
            if type(key) not in (str, int):
                raise ValueError("Selector keys must be literal strings or integers")
            keys.append(key)
            node = node.value
        if not isinstance(node, ast.Name) or node.id.startswith("__"):
            raise ValueError("Selector must start with an ordinary local")
        if len(keys) > 4:
            raise ValueError("Selector depth exceeds the language bound of four")
        return node.id, tuple(reversed(keys))
    except (SyntaxError, TypeError) as exc:
        raise ValueError("Invalid selection-only mapping") from exc


def locate(values: dict, selector: str):
    """Return the live parent container and final key, without copying values."""
    name, keys = parse(selector)
    container = values
    key: str | int = name
    for next_key in keys:
        container = container[key]
        if type(container) not in (list, dict):
            raise ValueError("Selectors traverse only lists and dictionaries")
        if type(container) is list and type(next_key) is not int:
            raise ValueError("List selection requires an integer index")
        key = next_key
    return container, key


def select(values: dict, selector: str):
    """Read exactly the selected current value."""
    container, key = locate(values, selector)
    return container[key]


def inventory(values: dict, depth: int, limit: int) -> tuple[dict, bool]:
    """Breadth-first, name-agnostic path inventory with bounded visits and depth."""
    queue = deque((name, value, 0) for name, value in sorted(values.items()))
    found: dict[str, Any] = {}
    while queue and len(found) < limit:
        path, value, level = queue.popleft()
        found[path] = value
        if level == depth:
            continue
        children: Iterable[tuple[str | int, Any]]
        if type(value) is dict:
            children = sorted(
                ((k, v) for k, v in value.items() if type(k) in (str, int)),
                key=lambda pair: (type(pair[0]).__name__, repr(pair[0])),
            )
        elif type(value) is list:
            children = enumerate(value)
        else:
            continue
        # No need to enqueue more children than this inventory can ever visit.
        for index, (key, child) in enumerate(children):
            if index >= limit:
                break
            queue.append((f"{path}[{json.dumps(key)}]", child, level + 1))
    return found, bool(queue)
