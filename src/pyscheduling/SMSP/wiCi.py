from dataclasses import dataclass, field
from typing import ClassVar, List

import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.core.base_solvers import BaseSolver
from pyscheduling.Problem import Objective
from pyscheduling.SMSP.SingleMachine import Constraints


@dataclass(init=False)
class wiCi_Instance(SingleMachine.SingleInstance):

    P: List[int]
    W: List[int]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.W]
    objective: ClassVar[Objective] = Objective.wiCi
    
    @property
    def init_sol_method(self):
        return WSPT()

class WSPT(BaseSolver):

    def solve(self, instance : wiCi_Instance):
        """Weighted Shortest Processing Time is Optimal for wiCi problem. A proof by contradiction can simply be found
        by performing an adjacent jobs interchange

        Args:
            instance (wiCi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method.
        """
        self.notify_on_start()
        jobs = list(range(instance.n))
        jobs.sort(reverse=True,key=lambda job_id : float(instance.W[job_id])/float(instance.P[job_id]))
        solution = SingleMachine.SingleSolution(instance)
        for job in jobs:
            solution.machine.job_schedule.append(SingleMachine.Job(job,0,0)) 
        solution.compute_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result