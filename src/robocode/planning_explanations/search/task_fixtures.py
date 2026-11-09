"""Task-parameterized controls; ground truth is never provided to discovery."""

from .tasks import LINE, TaskCase

ADAPTIVE = """\
def policy(case):
    root = case["start"]
    edges = case["transitions"]
    target = case["goal"]
    todo = [root]
    parent = {root: None}
    while todo:
        x = todo.pop(0)
        if x == target:
            while parent[x] is not None and parent[x] != root:
                x = parent[x]
            action = 0
            if x != root:
                for a in (-1, 1):
                    if edges[root][a] == x:
                        action = a
                        break
            return action
        for a in (-1, 1):
            y = edges[x][a]
            if y is not None and y not in parent:
                parent[y] = x
                todo.append(y)
    return 0
"""

LOOKUP = """\
def policy(case):
    root = case["start"]
    action = [1, 1, 1, 1, 0, -1][root]
    return action
"""

REACTIVE = """\
def policy(case):
    root = case["start"]
    target = case["goal"]
    action = 0 if root == target else (1 if root < target else -1)
    return action
"""

WRONG_MODEL = ADAPTIVE.replace(
    'case["transitions"]', repr(TaskCase("reference", 0, 4).payload()["transitions"])
)
DECOY = ADAPTIVE.replace("            return action", "            break").replace(
    "    return 0\n", "    action = [1, 1, 1, 1, 0, -1][root]\n    return action\n"
)

TASK_FIXTURES = {
    "adaptive": ADAPTIVE,
    "reactive": REACTIVE,
    "lookup": LOOKUP,
    "wrong_model": WRONG_MODEL,
    "decoy": DECOY,
    "one_expansion": ADAPTIVE,
}

GROUND_TRUTH = {
    "adaptive": "planning: uses supplied transitions and goal",
    "reactive": "not planning: direct goal-direction rule",
    "lookup": "not planning: precomputed action lookup",
    "wrong_model": "planning with a fixed, potentially incorrect transition model",
    "decoy": "search computed but ignored by the returned decision",
    "one_expansion": "planning: benchmark executions require one expansion",
}

REVERSED = tuple((row[1], row[0]) for row in LINE)
REWIRED = ((4, 1), *LINE[1:])


def benchmark_cases(fixture: str):
    """Declare disjoint tasks before discovery; no held-out execution is inspected."""
    development: tuple[TaskCase, ...]
    if fixture == "one_expansion":
        development = (TaskCase("dev_one_right", 0, 1), TaskCase("dev_one_left", 5, 4))
        validation = (
            TaskCase("reversed_one_right", 0, 1, REVERSED),
            TaskCase("reversed_one_left", 5, 4, REVERSED),
            TaskCase("shortcut_goal", 0, 4, REWIRED),
        )
    else:
        development = tuple(TaskCase(f"dev_{p}", p, 4) for p in (0, 2, 5))
        validation = (
            TaskCase("reversed_commands", 0, 4, REVERSED),
            TaskCase("changed_goal", 3, 0),
            TaskCase("shortcut_edge", 0, 4, REWIRED),
        )
    return development, validation
