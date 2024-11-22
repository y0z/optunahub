from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

import optuna

from optunahub.benchmarks import BaseProblem


class ConstrainedBaseProblem(BaseProblem):
    @abstractmethod
    def constraints_func(self, trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            trial: Optuna trial object.
        Returns:
            The constraint values.
        """
        ...
