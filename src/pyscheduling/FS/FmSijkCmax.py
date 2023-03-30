from dataclasses import dataclass
from typing import ClassVar, List

import pyscheduling.FS.FlowShop as FlowShop
import pyscheduling.FS.FS_methods as FS_methods
from pyscheduling.FS.FlowShop import Constraints
from pyscheduling.Problem import Objective


@dataclass(init=False)
class FmSijkCmax_Instance(FlowShop.FlowShopInstance):
    
    P: List[List[int]]
    S: List[List[List[int]]]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.S]
    objective: ClassVar[Objective] = Objective.Cmax

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.MINIT


class Heuristics(FS_methods.Heuristics):

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

class Metaheuristics(FS_methods.Metaheuristics):
    pass