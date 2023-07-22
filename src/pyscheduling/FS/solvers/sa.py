from dataclasses import dataclass, field
from typing import Callable

import pyscheduling.FS.FlowShop as FS
from pyscheduling.Problem import LocalSearch
from pyscheduling.core.base_solvers import BaseSA


@dataclass
class SA(BaseSA):

    ls_procedure: LocalSearch = field(default_factory = FS.FS_LocalSearch)
    generate_neighbour: Callable = field(default=FS.NeighbourhoodGeneration.random_neighbour)
    