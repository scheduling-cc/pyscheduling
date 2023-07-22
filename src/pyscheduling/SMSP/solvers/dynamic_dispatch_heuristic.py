
from dataclasses import dataclass
from functools import partial
from typing import Callable
import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.SMSP.SingleMachine import Job
from pyscheduling.core.base_solvers import BaseSolver


@dataclass
class DynamicDispatchHeuristic(BaseSolver):

    rule : Callable
    filter : Callable
    reverse : bool = False

    def solve(self, instance : SingleMachine.SingleInstance):
        """Orders the jobs respecting the filter according to the rule. 
        The order is dynamic since it is determined each time a new job is inserted

        Args:
            instance (SingleInstance): Instance to be solved
            rule (Callable): a lambda function that defines the sorting criteria taking the instance and job_id as the parameters
            filter (Callable): a lambda function that defines a filter condition taking the instance, job_id and current time as the parameters
            reverse (bool, optional): flag to sort in decreasing order. Defaults to False.

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
        solution = SingleMachine.SingleSolution(instance)

        remaining_jobs_list = list(range(instance.n))
        ci = min(instance.R)
        
        insert_idx = 0
        while(len(remaining_jobs_list)>0):
            ci = max( ci, min(instance.R[job_id] for job_id in remaining_jobs_list) ) # Advance the current ci to at least a time t > min_Ri
            filtered_remaining_jobs_list = list(filter(partial(self.filter_fun, instance, ci),remaining_jobs_list))
            filtered_remaining_jobs_list.sort(key= partial(self.rule, instance, ci), reverse=self.reverse)

            taken_job = filtered_remaining_jobs_list[0]
            #ci = solution.machine.objective_insert(taken_job, insert_idx, instance)
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,-1,-1))
            solution.compute_objective()
            ci = solution.objective_value
            remaining_jobs_list.remove(taken_job)
            insert_idx += 1
        
        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result