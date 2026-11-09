"""The independent physical task and weak existential trajectory witnesses."""

# Primitive IDs must be integers, not bools or user-defined numeric objects.
# pylint: disable=unidiomatic-typecheck

from collections import deque

from .engine import SearchSpec

DOMAIN = {"positions": 6, "actions": [-1, 0, 1], "goal_at_least": 4}


def move(position: int, action: int) -> int:
    """Execute a valid primitive line-world move."""
    if (
        type(position) is not int
        or not 0 <= position < 6
        or type(action) is not int
        or action not in (-1, 0, 1)
        or not 0 <= position + action < 6
    ):
        raise ValueError("Invalid physical transition")
    return position + action


def replay(start: int, actions: tuple[int, ...]) -> list[int]:
    """Replay a supplied nonempty witness using the physical environment."""
    if type(start) is not int or not 0 <= start < 6 or not actions:
        raise ValueError("A witness needs a valid start and a nonempty trajectory")
    trajectory = [start]
    for action in actions:
        trajectory.append(move(trajectory[-1], action))
    return trajectory


def check_environment(spec: SearchSpec, max_length: int = 3) -> dict:
    """Check goal meaning and one replayed witness per abstract action identity.

    Witnesses are not required for terminal action edges. Both families declare
    a task-goal predicate, although reactive search stops on Action nodes instead.
    """
    goals = []
    for position in range(6):
        predicted = spec.task_goal(spec.abstract(position))
        actual = position >= 4
        goals.append({"position": position, "abstract": predicted, "physical": actual})
        if predicted != actual:
            return {
                "status": "rejected",
                "reason": "goal_meaning",
                "counterexample": goals[-1],
            }
    witnesses = []
    if spec.mode == "graph":
        for abstract_action in (-1, 1):
            witness = None
            queue: deque[tuple[int, int, tuple[int, ...]]] = deque(
                (s, s, ()) for s in range(6)
            )
            while queue and witness is None:
                start, current, actions = queue.popleft()
                if len(actions) >= max_length:
                    continue
                for action in (-1, 0, 1):
                    try:
                        end = move(current, action)
                    except ValueError:
                        continue
                    path = actions + (action,)
                    x, y = spec.abstract(start), spec.abstract(end)
                    if y == x + abstract_action and y in spec.successors(x):
                        trajectory = replay(start, path)
                        witness = {
                            "action": abstract_action,
                            "start": start,
                            "actions": list(path),
                            "trajectory": trajectory,
                            "abstract_endpoints": [x, y],
                        }
                        break
                    queue.append((start, end, path))
            if witness is None:
                return {
                    "status": "unresolved",
                    "reason": "witness_budget",
                    "action": abstract_action,
                    "max_length": max_length,
                }
            witnesses.append(witness)
    return {
        "status": "supported",
        "goal_checks": goals,
        "witnesses": witnesses,
        "semantics": "One existential witness per abstract action, not per application",
        "max_witness_length": max_length,
        "attributed_policy_work": 0,
    }
