"""Nested representations and mixed routes; construction labels stay out of discovery."""

import ast

from .task_fixtures import ADAPTIVE, REVERSED, REWIRED
from .tasks import TaskCase


def nested(
    source: str,
    variable="work",
    keys=("payload", "root", "queue", "parents"),
    names=("root", "todo", "parent"),
) -> str:
    """Build equivalent fixtures by changing storage, never the search algorithm.

    ``keys[0]`` names the inner dictionary; ``keys[1:]`` rename ``names``.
    """
    if len(keys) != len(names) + 1:
        raise ValueError("Provide one container key plus one key per moved local")
    bindings = dict(zip(names, keys[1:]))

    class Pack(ast.NodeTransformer):
        """Move the named locals into two levels of dictionary storage."""

        def visit_Name(self, node):  # pylint: disable=invalid-name
            """Replace a local by a literal container path in the fixture only."""
            if node.id not in bindings:
                return node
            replacement = ast.parse(
                f"{variable}[{keys[0]!r}][{bindings[node.id]!r}]", mode="eval"
            ).body
            replacement.ctx = node.ctx
            return ast.copy_location(replacement, node)

    tree = Pack().visit(ast.parse(source))
    tree.body[0].body.insert(
        0, ast.parse(f"{variable} = {{{keys[0]!r}: {{}}}}").body[0]
    )
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


MIXED = ADAPTIVE.replace(
    '    root = case["start"]',
    '    if case["start"] >= 4:\n'
    '        action = 0 if case["start"] == case["goal"] else '
    '(1 if case["start"] < case["goal"] else -1)\n'
    "        return action\n"
    '    root = case["start"]',
)
UNSEEN = MIXED.replace(
    "def policy(case):\n",
    'def policy(case):\n    if case["start"] == 1:\n'
    "        action = 0\n        return action\n",
)

ROUTE_FIXTURES = {
    "nested": nested(ADAPTIVE),
    "nested_renamed": nested(ADAPTIVE, "z17", ("a9", "b8", "c7", "d6")),
    "mixed": MIXED,
    "mixed_nested": nested(MIXED),
    "unseen_route": UNSEEN,
}
GROUND_TRUTH = {
    "nested": "planning with nested dictionary storage",
    "nested_renamed": "same planning algorithm with renamed container and keys",
    "mixed": "search below start 4; direct rule otherwise",
    "mixed_nested": "same mixed policy with nested search storage",
    "unseen_route": "mixed policy plus a route absent from development",
}


def route_cases(fixture: str):
    """Include both known branches; change maps/goals in the held-out partition."""
    development = tuple(TaskCase(f"dev_{p}", p, 4) for p in (0, 2, 5)) + (
        TaskCase("dev_other_goal", 4, 5),
    )
    validation: tuple[TaskCase, ...] = (
        TaskCase("reversed_search", 0, 4, REVERSED),
        TaskCase("new_goal_search", 3, 0),
        TaskCase("reactive_left", 5, 0, REVERSED),
        TaskCase("reactive_at_goal", 4, 4, REWIRED),
    )
    if fixture == "unseen_route":
        validation += (TaskCase("unseen_prefix", 1, 4, REVERSED),)
    return development, validation
