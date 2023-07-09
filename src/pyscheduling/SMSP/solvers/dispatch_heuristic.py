
from dataclasses import dataclass
from functools import partial
from typing import Callable
import pyscheduling.Problem as Problem
import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.SMSP.SingleMachine import Job
from pyscheduling.core.base_solvers import BaseSolver


@dataclass
class DispatchHeuristic(BaseSolver):

    rule : Callable
    reverse : bool = False

    def solve(self, instance : SingleMachine.SingleInstance):
        """Orders the jobs according to the rule (lambda function) and returns the schedule accordignly

        Args:
            instance (SingleInstance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
        solution = SingleMachine.SingleSolution(instance)
        
        remaining_jobs_list = list(range(instance.n))
        sort_rule = partial(self.rule, instance)

        remaining_jobs_list.sort(key=sort_rule, reverse=self.reverse)
        solution.machine.job_schedule = [SingleMachine.Job(job_id, -1, -1) for job_id in remaining_jobs_list]
        solution.compute_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result