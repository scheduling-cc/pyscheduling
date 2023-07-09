from dataclasses import dataclass, field
from typing import Callable

import pyscheduling.SMSP.SingleMachine as sm
from pyscheduling.Problem import LocalSearch
from pyscheduling.core.base_solvers import BaseSA


@dataclass
class SA(BaseSA):

    ls_procedure: LocalSearch = field(default_factory = sm.SM_LocalSearch)
    generate_neighbour: Callable = field(default=sm.NeighbourhoodGeneration.lahc_neighbour)
    