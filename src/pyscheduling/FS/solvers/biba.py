from dataclasses import dataclass
import pyscheduling.FS.FlowShop as FS
from pyscheduling.Problem import Job
from pyscheduling.core.base_solvers import BaseSolver

@dataclass
class BIBA(BaseSolver):

    def solve(self, instance: FS.FlowShopInstance):
        """the greedy constructive heuristic (Best Insertion Based approach) to find an initial solution of flowshop instances 

        Args:
            instance (FlowShopInstance): Instance to be solved by the heuristic


        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        self.notify_on_start()
        solution = FS.FlowShopSolution(instance=instance)

        remaining_jobs_list = [j for j in range(instance.n)]

        while len(remaining_jobs_list) != 0:
            min_obj = None
            for i in remaining_jobs_list:

                start_time, end_time = solution.simulate_insert_last(i)
                new_obj = solution.simulate_insert_objective(i, start_time, end_time)

                if not min_obj or (min_obj > new_obj):
                    min_obj = new_obj
                    taken_job = i

            solution.job_schedule.append(Job(taken_job, 0, 0))
            solution.compute_objective(startIndex=len(solution.job_schedule) - 1)
            remaining_jobs_list.remove(taken_job)
        
        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result
