"""Data structures."""

from dataclasses import dataclass
from typing import Hashable, TypeAlias

# Define some data structures for this example repository (to be changed).
State: TypeAlias = Hashable
Action: TypeAlias = int
Goal: TypeAlias = str


@dataclass(frozen=True)
class Task:
    """An initial state and goal."""

    init: State
    goal: Goal
