import pyscheduling.PMSP.ParallelMachines as pm
from pyscheduling.Problem import Job
from pyscheduling.core.base_solvers import BaseSolver

class BIBA(BaseSolver):

    def solve(self, instance: pm.ParallelInstance):
        """the greedy constructive heuristic (Best insertion based approach) to find an initial solution of a PMSP.

        Args:
            instance (ParallelInstance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        self.notify_on_start()
        solution = pm.ParallelSolution(instance)
        remaining_jobs_list = [i for i in range(instance.n)]
        while len(remaining_jobs_list) != 0:
            min_factor = None
            for i in remaining_jobs_list:
                for j in range(instance.m):
                    current_machine = solution.machines[j]
                    last_pos = len(current_machine.job_schedule)
                    factor = current_machine.simulate_remove_insert(-1, i, last_pos, instance)
                    if min_factor is None or (min_factor > factor):
                        min_factor = factor
                        taken_job = i
                        taken_machine = j

            curr_machine = solution.machines[taken_machine]
            last_pos = len(curr_machine.job_schedule)
            curr_machine.job_schedule.append(Job(taken_job, -1, -1) )
            curr_machine.last_job = taken_job
            curr_machine.compute_objective(instance, startIndex=last_pos)
            remaining_jobs_list.remove(taken_job)
        
        solution.fix_objective()
        self.notify_on_solution_found(solution)
        self.notify_on_complete()
          
        return self.solve_result
