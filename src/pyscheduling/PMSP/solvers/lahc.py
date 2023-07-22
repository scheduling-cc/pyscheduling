from dataclasses import dataclass, field
from typing import Callable

import pyscheduling.PMSP.ParallelMachines as pm
from pyscheduling.Problem import LocalSearch
from pyscheduling.core.base_solvers import BaseLAHC


@dataclass
class LAHC(BaseLAHC):

    ls_procedure: LocalSearch = field(default_factory = pm.PM_LocalSearch)
    generate_neighbour: Callable = field(default=pm.NeighbourhoodGeneration.lahc_neighbour)
