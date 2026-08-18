from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Sequence
import warnings

import optuna


def _as_constraints_dict(constraints: dict[str, float] | Sequence[float]) -> dict[str, float]:
    if isinstance(constraints, dict):
        return constraints
    # `evaluate_constraints` used to return a sequence. Such a return value is still accepted
    # and each value is named by its index, which is the same naming as the one Optuna uses
    # for the constraints stored in the old format.
    warnings.warn(
        "Returning a sequence from `evaluate_constraints` is deprecated in v0.5.0. "
        "The benchmark package cached in your local directory is outdated. "
        "Please update it by `optunahub.load_module(..., force_reload=True)`.",
        FutureWarning,
    )
    return {str(i): value for i, value in enumerate(constraints)}


class BaseProblem(metaclass=ABCMeta):
    """Base class for optimization problems."""

    def __call__(self, trial: optuna.Trial) -> float | Sequence[float]:
        """Objective function for Optuna. By default, this method calls :meth:`evaluate` with the parameters defined in :attr:`search_space`.

        Args:
            trial: Optuna trial object.
        Returns:
            The objective value or a sequence of the objective values for multi-objective optimization.
        """
        params = {}
        for name, dist in self.search_space.items():
            params[name] = trial._suggest(name, dist)
            trial._check_distribution(name, dist)
        constraints = _as_constraints_dict(self.evaluate_constraints(params))
        if constraints and not hasattr(trial, "set_constraint"):
            raise RuntimeError("`evaluate_constraints` requires Optuna v5.0.0 or newer.")
        for key, value in constraints.items():
            trial.set_constraint(key, value)
        return self.evaluate(params)

    def evaluate(self, params: dict[str, Any]) -> float | Sequence[float]:
        """Evaluate the objective function.

        Args:
            params: Dictionary of input parameters.

        Returns:
            The objective value or a sequence of the objective values for multi-objective optimization.

        Example:
            ::

                def evaluate(self, params: dict[str, Any]) -> float:
                    x = params["x"]
                    y = params["y"]
                    return x ** 2 + y
        """
        raise NotImplementedError

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space.

        Returns:
            Dictionary of search space. Each dictionary element consists of the parameter name and distribution (see `optuna.distributions <https://optuna.readthedocs.io/en/stable/reference/distributions.html>`__).

        Example:
            ::

                @property
                def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
                    return {
                        "x": optuna.distributions.FloatDistribution(low=0, high=1),
                        "y": optuna.distributions.CategoricalDistribution(choices=[0, 1, 2]),
                    }
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions.

        Returns:
            List of `optuna.study.StudyDirection <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.StudyDirection.html>`__.

        Example:
            ::

                @property
                def directions(self) -> list[optuna.study.StudyDirection]:
                    return [optuna.study.StudyDirection.MINIMIZE]
        """
        ...

    def evaluate_constraints(self, params: dict[str, Any]) -> dict[str, float]:
        """Evaluate the constraint functions.

        Args:
            params: Dictionary of input parameters.
        Returns:
            Dictionary of the constraint values keyed by the constraint names.
            A trial is considered feasible when all the values are zero or less.
        """
        return {}

    def constraints_func(self, trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        """Evaluate the constraint functions.

        .. warning::
            Deprecated in v0.5.0. This feature will be removed in the future without prior notice.

        Args:
            trial: Optuna trial object.
        Returns:
            List of the constraint values.
        """
        warnings.warn(
            "`constraints_func` is deprecated in v0.5.0 since OptunaHub's problems now set the "
            "constraint values to each trial by themselves. Please stop passing "
            "`constraints_func` to your sampler.",
            FutureWarning,
        )

        constraints = self.evaluate_constraints(trial.params.copy())  # type: ignore[attr-defined]
        return list(_as_constraints_dict(constraints).values())
