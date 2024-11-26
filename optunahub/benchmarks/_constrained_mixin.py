from typing import Any
from typing import Sequence

import optuna


class ConstrainedMixIn:
    def constraints_func(self, trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            trial: Optuna trial object.
        Returns:
            The constraint values.
        """
        return self.evaluate_constraints(trial.params)

    def evaluate_constraints(self, params: dict[str, Any]) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            params: Input vector.
        Returns:
            The constraint values.
        """
        return self.constraints_func(optuna.trial.FixedTrial(params))
