from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Sequence

import optuna


class BaseProblem(metaclass=ABCMeta):
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

    @property
    @abstractmethod
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        ...

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        raise NotImplementedError

    def evaluate(self, params: dict[str, Any]) -> float | Sequence[float]:
        """Evaluate the objective function.
        Args:
            params: Dictionary of input parameters.
        Returns:
            The objective value.
        """
        raise NotImplementedError
