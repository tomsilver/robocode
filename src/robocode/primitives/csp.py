"""Constraint satisfaction problem (CSP) primitive."""

import abc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm import tqdm  # type: ignore[import-untyped]


@dataclass(frozen=True)
class CSPVariable:
    """Constraint satisfaction problem variable."""

    name: str
    domain: gym.spaces.Space

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, CSPVariable)
        return self.name == other.name and str(self.domain) == str(other.domain)


class CSPConstraint(abc.ABC):
    """Constraint satisfaction problem constraint."""

    def __init__(self, name: str, variables: list[CSPVariable]):
        self.name = name
        self.variables = variables

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.variables)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CSPConstraint):
            return False
        return self.name == other.name and self.variables == other.variables

    @abc.abstractmethod
    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        """Check whether the constraint holds given values of the variables."""

    @abc.abstractmethod
    def copy(self) -> "CSPConstraint":
        """Create a copy of this constraint."""


class FunctionalCSPConstraint(CSPConstraint):
    """A constraint defined by a function that outputs bools."""

    def __init__(
        self,
        name: str,
        variables: list[CSPVariable],
        constraint_fn: Callable[..., bool],
    ):
        super().__init__(name, variables)
        self.constraint_fn = constraint_fn
        self._cache: dict[tuple, bool] = {}

    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        vals = tuple(sol[v] for v in self.variables)
        try:
            return self._cache[vals]
        except (TypeError, KeyError):
            pass
        result = self.constraint_fn(*vals)
        try:
            self._cache[vals] = result
        except TypeError:
            pass
        return result

    def copy(self) -> "CSPConstraint":
        return FunctionalCSPConstraint(self.name, self.variables, self.constraint_fn)


class LogProbCSPConstraint(CSPConstraint):
    """A constraint defined by a function that outputs log probabilities.

    The constraint holds when the log probability is above the threshold.
    """

    def __init__(
        self,
        name: str,
        variables: list[CSPVariable],
        constraint_logprob_fn: Callable[..., float],
        threshold: float = np.log(0.95),
    ):
        super().__init__(name, variables)
        self.constraint_logprob_fn = constraint_logprob_fn
        self.threshold = threshold

    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        return self.get_logprob(sol) >= self.threshold

    def get_logprob(self, sol: dict[CSPVariable, Any]) -> float:
        """Get the log probability of the constraint holding."""
        vals = [sol[v] for v in self.variables]
        return self.constraint_logprob_fn(*vals)

    def copy(self) -> "CSPConstraint":
        return LogProbCSPConstraint(
            self.name, self.variables, self.constraint_logprob_fn, self.threshold
        )


@dataclass(frozen=True)
class CSPCost:
    """A cost function to be minimized over certain CSP variables."""

    name: str
    variables: list[CSPVariable]
    cost_fn: Callable[..., float]

    def get_cost(self, sol: dict[CSPVariable, Any]) -> float:
        """Evaluate the cost function."""
        vals = [sol[v] for v in self.variables]
        return self.cost_fn(*vals)


@dataclass(frozen=True)
class CSP:
    """Constraint satisfaction problem."""

    variables: list[CSPVariable]
    constraints: list[CSPConstraint]
    cost: CSPCost | None = None

    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        """Check whether all constraints hold given values of the variables."""
        for constraint in self.constraints:
            logging.debug(f"Checking constraint: {constraint.name}")
            if not constraint.check_solution(sol):
                logging.debug("Result: False")
                return False
            logging.debug("Result: True")
        return True

    def get_cost(self, sol: dict[CSPVariable, Any]) -> float:
        """Evaluate the cost function."""
        assert self.cost is not None
        return self.cost.get_cost(sol)


class CSPSampler(abc.ABC):
    """Samples values of one or more variables in a CSP.

    The sampler can optionally use existing bindings of variables, e.g., for conditional
    sampling, or for MCMC-style sampling.
    """

    def __init__(self, csp: CSP, sampled_vars: set[CSPVariable]) -> None:
        assert sampled_vars.issubset(csp.variables)
        self._csp = csp
        self._sampled_vars = sampled_vars

    @abc.abstractmethod
    def sample(
        self, current_vals: dict[CSPVariable, Any], rng: np.random.Generator
    ) -> dict[CSPVariable, Any] | None:
        """Sample values for self.sampled_vars given values of all CSP vars."""


class FunctionalCSPSampler(CSPSampler):
    """A CSPSampler implemented with a function."""

    def __init__(
        self,
        fn: Callable[
            [dict[CSPVariable, Any], np.random.Generator],
            dict[CSPVariable, Any] | None,
        ],
        csp: CSP,
        sampled_vars: set[CSPVariable],
    ) -> None:
        self._fn = fn
        super().__init__(csp, sampled_vars)

    def sample(
        self, current_vals: dict[CSPVariable, Any], rng: np.random.Generator
    ) -> dict[CSPVariable, Any] | None:
        return self._fn(current_vals, rng)


class CSPSolver(abc.ABC):
    """A CSP solver."""

    def __init__(self, seed: int) -> None:
        self._seed = seed

    @abc.abstractmethod
    def solve(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        """Solve the given CSP."""


class RandomWalkCSPSolver(CSPSolver):
    """Random walk solver that remembers the best satisfying solution."""

    def __init__(
        self,
        seed: int,
        max_iters: int = 100_000,
        num_improvements: int = 5,
        max_improvement_attempts: int = 1_000,
        show_progress_bar: bool = True,
    ) -> None:
        super().__init__(seed)
        self._max_iters = max_iters
        self._num_improvements = num_improvements
        self._max_improvement_attempts = max_improvement_attempts
        self._show_progress_bar = show_progress_bar
        self._rng = np.random.default_rng(seed)

    def solve(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        sol = initialization.copy()
        best_satisfying_sol: dict[CSPVariable, Any] | None = None
        best_satisfying_cost: float = np.inf
        solution_found = False
        num_improve_attempts = 0
        num_improve_found = 0
        sampler_idxs = list(range(len(samplers)))
        for _ in (
            pbar := tqdm(range(self._max_iters), disable=not self._show_progress_bar)
        ):
            if solution_found and (
                num_improve_attempts >= self._max_improvement_attempts
                or num_improve_found >= self._num_improvements
            ):
                break
            if solution_found:
                num_improve_attempts += 1
                msg = (
                    f"Improved {num_improve_found} times w/ "
                    f"{num_improve_attempts} tries)"
                )
            else:
                msg = "Searching for first solution"
            pbar.set_description(msg)

            sol_is_cost_improvement = True
            if csp.cost is not None:
                cost = csp.get_cost(sol)
                if cost > best_satisfying_cost:
                    sol_is_cost_improvement = False

            if sol_is_cost_improvement and csp.check_solution(sol):
                if solution_found:
                    num_improve_found += 1
                solution_found = True
                if csp.cost is None:
                    return sol
                best_satisfying_cost = cost
                best_satisfying_sol = sol

            self._rng.shuffle(sampler_idxs)
            for sample_idx in sampler_idxs:
                sampler = samplers[sample_idx]
                partial_sol = sampler.sample(sol, self._rng)
                if partial_sol is not None:
                    break
            else:
                raise RuntimeError("All samplers produced None; solver stuck.")
            sol = sol.copy()
            sol.update(partial_sol)
        return best_satisfying_sol


__all__ = [
    "CSP",
    "CSPConstraint",
    "CSPCost",
    "CSPSampler",
    "CSPSolver",
    "CSPVariable",
    "FunctionalCSPConstraint",
    "FunctionalCSPSampler",
    "LogProbCSPConstraint",
    "RandomWalkCSPSolver",
]
