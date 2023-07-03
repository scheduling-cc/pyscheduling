from dataclasses import dataclass, field
from time import perf_counter
from typing import ClassVar, List

import pyscheduling.FS.FlowShop as FlowShop
from pyscheduling.FS.solvers import BIBA
import pyscheduling.Problem as Problem
from pyscheduling.FS.FlowShop import Constraints
from pyscheduling.Problem import Job, Objective
from pyscheduling.core.base_solvers.base_solver import BaseSolver


@dataclass
class FmCmax_Instance(FlowShop.FlowShopInstance):
    P: List[List[int]] = field(default_factory=list)  # Processing time
    constraints: ClassVar[List[Constraints]] = [Constraints.P]
    objective: ClassVar[Objective] = Objective.Cmax
    init_sol_method: BaseSolver = field(default_factory=BIBA)


@dataclass
class Slope(BaseSolver):

    def solve(self, instance: FmCmax_Instance):
        """Inspired from Jonhson's rule, this heuristic schedules first the jobs with the smallest processing times on the first machines

        Args:
            instance (FmCmax_Instance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        self.notify_on_start()
        solution = FlowShop.FlowShopSolution(instance=instance)
        jobs = list(range(instance.n))
        # m+1 to translate the set of numbers of m from [[0,m-1]] to [[1,m]]
        # machine_id+1 to translate the set of numbers of machine_id from [[0,m-1]] to [[1,m]]
        slope_index = lambda job_id : -sum([((instance.m + 1) - (2*(machine_id+1)-1))*instance.P[job_id][machine_id] for machine_id in range(instance.m)])
        jobs.sort(reverse=True,key=slope_index)
        solution.job_schedule = [ Job(job_id, 0, 0) for job_id in jobs]
        solution.compute_objective()

        self.notify_on_solution_found()
        self.notify_on_complete()

        return self.solve_result