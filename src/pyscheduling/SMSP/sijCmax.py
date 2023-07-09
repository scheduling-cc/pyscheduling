from dataclasses import dataclass, field
from typing import ClassVar, List

import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.Problem import Objective
from pyscheduling.SMSP.SingleMachine import Constraints
from pyscheduling.SMSP.solvers import BIBA
from pyscheduling.core.base_solvers import BaseSolver


@dataclass(init=False)
class sijCmax_Instance(SingleMachine.SingleInstance):

    P: List[int]
    S: List[List[int]]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.S]
    objective: ClassVar[Objective] = Objective.Cmax
    init_sol_method: BaseSolver = BIBA()
