from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Sequence

import optuna


class BaseProblem(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, trial: optuna.Trial) -> float | Sequence[float]:
        """Objective function for Optuna.
        Args:
            trial: Optuna trial object.
        Returns:
            The objective value.
        """
        ...

    @property
    def direction(self) -> optuna.study.StudyDirection:
        """Return the optimization direction."""
        if len(self.directions) > 1:
            raise ValueError("This problem has multiple directions.")
        return self.directions[0]

    @property
    @abstractmethod
    def directions(self) -> Sequence[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        ...
