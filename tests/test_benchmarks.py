from __future__ import annotations

import optuna

import optunahub


class TestProblem(optunahub.benchmarks.BaseProblem):
    def evaluate(self, params: dict[str, float]) -> float:
        x = params["x"]
        return x**2

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {"x": optuna.distributions.FloatDistribution(low=-1, high=1)}

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MINIMIZE]


def test_base_problem() -> None:
    problem = TestProblem()
    study = optuna.create_study(directions=problem.directions)
    study.optimize(problem, n_trials=20)  # verify no error occurs


def test_constrained_problem() -> None:
    class ConstrainedTestProblem(TestProblem):
        def evaluate_constraints(self, params: dict[str, float]) -> dict[str, float]:
            return {"c0": params["x"]}

    problem = ConstrainedTestProblem()
    study = optuna.create_study(directions=problem.directions)
    study.optimize(problem, n_trials=20)  # verify no error occurs

    # Check if constraints are stored in trials
    for t in study.trials:
        assert t.constraints == {"c0": t.params["x"]}
