from dataclasses import dataclass, field
from typing import ClassVar, List

import pyscheduling.FS.FlowShop as FlowShop
from pyscheduling.FS.FlowShop import Constraints
from pyscheduling.Problem import Objective
from pyscheduling.FS.solvers import BIBA
from pyscheduling.core.base_solvers import BaseSolver


@dataclass(init=False)
class FmriSijkwiCi_Instance(FlowShop.FlowShopInstance):
    
    P: List[List[int]]
    W: List[int]
    R: List[int]
    S: List[List[List[int]]]
    constraints: ClassVar[Constraints] = [Constraints.P, Constraints.W, Constraints.R, Constraints.S]
    objective: ClassVar[Objective] = Objective.wiCi
    init_sol_method: BaseSolver = BIBA()
