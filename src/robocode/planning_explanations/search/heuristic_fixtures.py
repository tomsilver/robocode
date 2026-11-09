"""Inspectable priority-search controls on a six-state merging-path graph."""

from textwrap import indent

from .heuristics import ZERO, HeuristicCase
from .route_fixtures import nested

# 0 --(-1)--> 1 --> 3 --> 4 --> 5: four edges
# 0 --(+1)--> 2 --------> 4 --> 5: three edges
FUNNEL = ((1, 2), (3, None), (4, None), (4, None), (5, None), (None, None))
REOPEN_H = (0, 0, 2, 0, 0, 0)
REVERSED = tuple((right, left) for left, right in FUNNEL)


def policy_source(
    selection: str = "astar",
    timing: str = "selection",
    relaxation: str = "improve",
    tie: str = "last",
) -> str:
    """Generate readable single-function fixture source, not verifier behavior."""
    readout = """\
z = found
while parent[z] is not None and parent[z] != root:
    z = parent[z]
action = 0
if z != root:
    for a in (-1, 1):
        if edges[root][a] == z:
            action = a
            break
return action
"""
    source = """\
def policy(case):
    root = case["start"]
    edges = case["transitions"]
    target = case["goal"]
    h = case["heuristic"]
    todo = [root]
    parent = {root: None}
    distance = {root: 0}
    while todo:
"""
    if selection in ("fifo", "lifo"):
        source += f"        index = {0 if selection == 'fifo' else -1}\n"
    else:
        score = (
            "h[todo[i]]" if selection == "greedy" else "distance[todo[i]] + h[todo[i]]"
        )
        best = score.replace("[i]", "[index]")
        operator = "<=" if tie == "last" else "<"
        source += (
            "        index = 0\n        for i in range(1, len(todo)):\n"
            f"            if {score} {operator} {best}:\n                index = i\n"
        )
    source += (
        "        x = todo.pop(index)\n        if x == target:\n            found = x\n"
    )
    source += indent(readout, "            ")
    condition = "y not in parent"
    if relaxation == "improve":
        condition += " or candidate < distance[y]"
    source += (
        "        for a in (-1, 1):\n            y = edges[x][a]\n"
        "            if y is not None:\n                candidate = distance[x] + 1\n"
        f"                if {condition}:\n"
        "                    parent[y] = x\n"
        "                    distance[y] = candidate\n"
        "                    if y not in todo:\n                        todo.append(y)\n"
    )
    if timing == "generation":
        source += (
            "                    if y == target:\n                        found = y\n"
        )
        source += indent(readout, "                        ")
    return source + "    return 0\n"


ASTAR = policy_source()
IGNORED_H = policy_source("fifo").replace(
    "        index = 0\n",
    "        priority = distance[todo[0]] + h[todo[0]]\n        index = 0\n",
)
UNUSED = ASTAR.replace(
    "            return action\n", "            action = 0\n            return action\n"
)
HEURISTIC_FIXTURES = {
    "astar_reopen": ASTAR,
    "greedy": policy_source("greedy"),
    "goal_generation": policy_source(timing="generation"),
    "first_parent": policy_source(relaxation="first"),
    "nested_astar": nested(ASTAR, "z17", ("a9", "b8", "c7", "d6")),
    # Also moves g ("distance") and h, so no priority-search role keeps its name.
    "nested_costs": nested(
        ASTAR,
        "z17",
        ("a9", "b8", "c7", "d6", "e5", "f4"),
        ("root", "todo", "parent", "distance", "h"),
    ),
    "ignored_heuristic": IGNORED_H,
    "unused_search": UNUSED,
}
GROUND_TRUTH = {
    "astar_reopen": (
        "A* with strict cost improvement, reopening and last-in-frontier ties"
    ),
    "greedy": "greedy heuristic selection with strict cost improvement",
    "goal_generation": "A* ranking but early goal-on-generation stopping",
    "first_parent": "A* ranking with first-parent retention; no cost improvement",
    "nested_astar": "same reopening A* with renamed nested search storage",
    "nested_costs": "same reopening A* with cost and heuristic tables also nested",
    "ignored_heuristic": "FIFO search; computed heuristic score is ignored",
    "unused_search": "search computed but its decision is replaced by a constant",
}


def heuristic_cases() -> tuple[tuple[HeuristicCase, ...], tuple[HeuristicCase, ...]]:
    """Declare task/heuristic shifts before discovery; preserve full input identity."""
    development = (
        HeuristicCase("dev_reopen", 0, 5, FUNNEL, REOPEN_H),
        HeuristicCase("dev_other_ranking", 0, 5, FUNNEL, (0, 5, 0, 2, 0, 0)),
        HeuristicCase("dev_join", 1, 5, FUNNEL, REOPEN_H),
    )
    validation = (
        HeuristicCase("reversed_commands", 0, 5, REVERSED, REOPEN_H),
        HeuristicCase("zero_heuristic", 0, 5, FUNNEL, ZERO),
        HeuristicCase("changed_goal", 0, 4, FUNNEL, REOPEN_H),
        HeuristicCase("already_at_goal", 5, 5, REVERSED, REOPEN_H),
    )
    return development, validation
