from dataclasses import dataclass
from typing import ClassVar, List

import pyscheduling.FS.FlowShop as FlowShop
import pyscheduling.FS.FS_methods as FS_methods
from pyscheduling.FS.FlowShop import Constraints
from pyscheduling.Problem import Objective


@dataclass(init=False)
class FmriSijkwiCi_Instance(FlowShop.FlowShopInstance):
    
    P: List[List[int]]
    W: List[int]
    R: List[int]
    S: List[List[List[int]]]
    constraints: ClassVar[Constraints] = [Constraints.P, Constraints.W, Constraints.R, Constraints.S]
    objective: ClassVar[Objective] = Objective.wiCi

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.BIBA


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