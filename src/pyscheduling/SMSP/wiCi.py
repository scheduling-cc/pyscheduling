from dataclasses import dataclass
from time import perf_counter
from typing import ClassVar, List

import pyscheduling.Problem as Problem
import pyscheduling.SMSP.SingleMachine as SingleMachine
import pyscheduling.SMSP.SM_methods as Methods
from pyscheduling.Problem import Objective
from pyscheduling.SMSP.SingleMachine import Constraints
from pyscheduling.SMSP.SM_methods import ExactSolvers


@dataclass(init=False)
class wiCi_Instance(SingleMachine.SingleInstance):

    P: List[int]
    W: List[int]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.W]
    objective: ClassVar[Objective] = Objective.wiCi

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.BIBA

class Heuristics(Methods.Heuristics):
    @staticmethod
    def WSPT(instance : wiCi_Instance):
        """Weighted Shortest Processing Time is Optimal for wiCi problem. A proof by contradiction can simply be found
        by performing an adjacent jobs interchange

        Args:
            instance (wiCi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method.
        """
        startTime = perf_counter()
        jobs = list(range(instance.n))
        jobs.sort(reverse=True,key=lambda job_id : float(instance.W[job_id])/float(instance.P[job_id]))
        solution = SingleMachine.SingleSolution(instance)
        for job in jobs:
            solution.machine.job_schedule.append(SingleMachine.Job(job,0,0)) 
        solution.compute_objective()
        return Problem.SolveResult(best_solution=solution,status=Problem.SolveStatus.OPTIMAL,runtime=perf_counter()-startTime,solutions=[solution])

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]


class Metaheuristics(Methods.Metaheuristics):
    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

