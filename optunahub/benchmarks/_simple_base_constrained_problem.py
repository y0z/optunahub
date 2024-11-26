from ._constrained_mixin import ConstrainedMixIn
from ._simple_base_problem import SimpleBaseProblem


class SimpleBaseConstrainedProblem(SimpleBaseProblem, ConstrainedMixIn):
    pass
