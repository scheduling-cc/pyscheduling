from dataclasses import dataclass
from typing import ClassVar, List

import pyscheduling.JS.JobShop as JobShop
from pyscheduling.core.base_solvers import BaseSolver
from pyscheduling.JS.JobShop import Constraints
from pyscheduling.JS.solvers import BIBA
from pyscheduling.Problem import Objective


@dataclass(init=False)
class JmridiSijkwiTi_Instance(JobShop.JobShopInstance):

    P: List[List[int]]
    W: List[int]
    R: List[int]
    D: List[int]
    S: List[List[List[int]]]
    constraints: ClassVar[Constraints] = [Constraints.P, Constraints.W, Constraints.R, Constraints.D, Constraints.S]
    objective: ClassVar[Objective] = Objective.wiTi
    init_sol_method: BaseSolver = BIBA()
    