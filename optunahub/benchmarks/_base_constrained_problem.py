from ._base_problem import BaseProblem
from ._constrained_mixin import ConstrainedMixIn


class BaseConstrainedProblem(BaseProblem, ConstrainedMixIn):
    pass
