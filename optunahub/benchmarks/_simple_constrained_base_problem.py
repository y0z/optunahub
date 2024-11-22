from __future__ import annotations

from abc import abstractmethod
from typing import Any
from typing import Sequence

import optuna

from optunahub.benchmarks import ConstrainedBaseProblem
from optunahub.benchmarks import SimpleBaseProblem


class SimpleConstrainedBaseProblem(ConstrainedBaseProblem, SimpleBaseProblem):
    def constraints_func(self, trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            trial: Optuna trial object.
        Returns:
            The constraint values.
        """
        return self.evaluate_constraints(trial.params)

    @abstractmethod
    def evaluate_constraints(self, params: dict[str, Any]) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            params: Input vector.
        Returns:
            The constraint values.
        """
        ...
