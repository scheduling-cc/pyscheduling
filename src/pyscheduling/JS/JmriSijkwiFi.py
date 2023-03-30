from dataclasses import dataclass
from typing import ClassVar, List

import pyscheduling.JS.JobShop as JobShop
import pyscheduling.JS.JS_methods as js_methods
from pyscheduling.JS.JobShop import Constraints
from pyscheduling.Problem import Objective


@dataclass(init=False)
class JmriSijkwiFi_Instance(JobShop.JobShopInstance):

    P: List[List[int]]
    W: List[int]
    R: List[int]
    S: List[List[List[int]]]
    constraints: ClassVar[Constraints] = [Constraints.P, Constraints.W, Constraints.R, Constraints.S]
    objective: ClassVar[Objective] = Objective.wiFi

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.BIBA
    

class Heuristics(js_methods.Heuristics):

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]