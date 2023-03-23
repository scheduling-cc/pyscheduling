from dataclasses import dataclass, field
from time import perf_counter
from typing import ClassVar, List

import pyscheduling.FS.FlowShop as FlowShop
import pyscheduling.FS.FS_methods as FS_methods
import pyscheduling.Problem as Problem
from pyscheduling.FS.FlowShop import Constraints
from pyscheduling.Problem import Job, Objective


@dataclass(init=False)
class FmCmax_Instance(FlowShop.FlowShopInstance):
    P: List[List[int]] = field(default_factory=list)  # Processing time
    constraints: ClassVar[List[Constraints]] = [Constraints.P]
    objective: ClassVar[Objective] = Objective.Cmax

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.slope


class Heuristics(FS_methods.Heuristics):

    @staticmethod
    def slope(instance: FmCmax_Instance):
        """Inspired from Jonhson's rule, this heuristic schedules first the jobs with the smallest processing times on the first machines

        Args:
            instance (FmCmax_Instance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = FlowShop.FlowShopSolution(instance=instance)
        jobs = list(range(instance.n))
        # m+1 to translate the set of numbers of m from [[0,m-1]] to [[1,m]]
        # machine_id+1 to translate the set of numbers of machine_id from [[0,m-1]] to [[1,m]]
        slope_index = lambda job_id : -sum([((instance.m + 1) - (2*(machine_id+1)-1))*instance.P[job_id][machine_id] for machine_id in range(instance.m)])
        jobs.sort(reverse=True,key=slope_index)
        solution.job_schedule = [ Job(job_id, 0, 0) for job_id in jobs]
        solution.compute_objective()
        return Problem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

class Metaheuristics(FS_methods.Metaheuristics):
    pass