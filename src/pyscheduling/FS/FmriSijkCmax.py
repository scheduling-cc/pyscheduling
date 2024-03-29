from dataclasses import dataclass
from typing import ClassVar, List

import pyscheduling.FS.FlowShop as FlowShop
from pyscheduling.FS.FlowShop import Constraints
from pyscheduling.Problem import Objective
from pyscheduling.FS.solvers import BIBA
from pyscheduling.core.base_solvers import BaseSolver


@dataclass(init=False)
class FmriSijkCmax_Instance(FlowShop.FlowShopInstance):

    P: List[List[int]]
    R: List[int]
    S: List[List[List[int]]]
    constraints: ClassVar[Constraints] = [Constraints.P, Constraints.R, Constraints.S]
    objective: ClassVar[Objective] = Objective.Cmax
    init_sol_method: BaseSolver = BIBA()


