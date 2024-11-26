from __future__ import annotations

from typing import Any
from typing import Sequence

import optuna

from ._base_problem import BaseProblem


class BaseConstrainedProblem(BaseProblem):
    def constraints_func(self, trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            trial: Optuna trial object.
        Returns:
            The constraint values.
        """
        return self.evaluate_constraints(trial.prams)

    def evaluate_constraints(self, params: dict[str, Any]) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            params: Input vector.
        Returns:
            The constraint values.
        """
        raise NotImplementedError
