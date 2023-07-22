from dataclasses import dataclass, field
from typing import ClassVar, List

import pyscheduling.FS.FlowShop as FlowShop
from pyscheduling.FS.FlowShop import Constraints
from pyscheduling.Problem import Objective
from pyscheduling.FS.solvers import MINIT
from pyscheduling.core.base_solvers import BaseSolver


@dataclass(init=False)
class FmSijkCmax_Instance(FlowShop.FlowShopInstance):
    
    P: List[List[int]]
    S: List[List[List[int]]]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.S]
    objective: ClassVar[Objective] = Objective.Cmax
    init_sol_method: BaseSolver = MINIT()

