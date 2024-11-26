from typing import Any
from typing import Sequence

import optuna


ERR_MSG = (
    "You need to reimplement one of the `constraints` or `evaluate_constraints` methods "
    "in the derived class since the default implementations have a mutual recursion."
)


class ConstrainedMixIn:
    """Mixin class for constrained optimization.
    This class provides the constraint functions.

    You need to reimplement one of the `constraints` or `evaluate_constraints` methods
    in the derived class since the default implementations have a mutual recursion.
    """

    def constraints_func(self, trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            trial: Optuna trial object.
        Returns:
            The constraint values.
        """
        try:
            return self.evaluate_constraints(trial.params)
        except RecursionError as err:
            raise NotImplementedError(ERR_MSG) from err

    def evaluate_constraints(self, params: dict[str, Any]) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            params: Input vector.
        Returns:
            The constraint values.
        """
        try:
            return self.constraints_func(optuna.trial.FixedTrial(params))
        except RecursionError as err:
            raise NotImplementedError(ERR_MSG) from err
