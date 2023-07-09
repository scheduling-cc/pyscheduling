from dataclasses import dataclass
from typing import ClassVar, List

import pyscheduling.JS.JobShop as JobShop
from pyscheduling.JS.JobShop import Constraints
from pyscheduling.Problem import Objective
from pyscheduling.core.base_solvers import BaseSolver
from pyscheduling.JS.solvers import BIBA

@dataclass(init=False)
class JmriSijkCmax_Instance(JobShop.JobShopInstance):

    P: List[List[int]]
    R: List[int]
    S: List[List[List[int]]]
    constraints: ClassVar[Constraints] = [Constraints.P, Constraints.R, Constraints.S]
    objective: ClassVar[Objective] = Objective.Cmax
    init_sol_method: BaseSolver = BIBA()
    