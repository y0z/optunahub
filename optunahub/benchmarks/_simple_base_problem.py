from __future__ import annotations

from abc import abstractmethod
from typing import Any
from typing import Sequence

import optuna

from optunahub.benchmarks import BaseProblem


class SimpleBaseProblem(BaseProblem):
    def __call__(self, trial: optuna.Trial) -> float | Sequence[float]:
        """Objective function for Optuna.
        Args:
            trial: Optuna trial object.
        Returns:
            The objective value.
        """
        params = {}
        for name, dist in self.search_space.items():
            params[name] = trial._suggest(name, dist)
            trial._check_distribution(name, dist)
        return self.evaluate(params)

    def constraints_func(self, trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            trial: Optuna trial object.
        Returns:
            The constraint values.
        """
        return self.evaluate_constraints(trial.params)

    @property
    @abstractmethod
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        ...

    @abstractmethod
    def evaluate(self, params: dict[str, Any]) -> float | Sequence[float]:
        """Evaluate the objective function.
        Args:
            params: Input vector.
        Returns:
            The objective value.
        """
        ...

    def evaluate_constraints(self, params: dict[str, Any]) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            params: Input vector.
        Returns:
            The constraint values.
        """
        raise NotImplementedError
