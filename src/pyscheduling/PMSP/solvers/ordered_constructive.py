
from typing import List
from dataclasses import dataclass
import random
import pyscheduling.PMSP.ParallelMachines as pm
from pyscheduling.core.base_solvers import BaseSolver


@dataclass
class OrderedConstructive(BaseSolver):
    
    remaining_jobs_list: List = None
    is_random: bool = False

    def solve(self, instance: pm.ParallelInstance):
        """the ordered greedy constructive heuristic to find an initial solution of RmSijkCmax problem minimalizing the factor of (processing time + setup time) of
        jobs in the given order on different machines

        Args:
            instance (pm.ParallelInstance): Instance to be solved by the heuristic
            
        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        self.notify_on_start()
        solution = pm.ParallelSolution(instance)
        if self.remaining_jobs_list is None:
            self.remaining_jobs_list = [i for i in range(instance.n)]
            if self.is_random:
                random.shuffle(self.remaining_jobs_list)
                
        for i in self.remaining_jobs_list:
            min_factor = None
            for j in range(instance.m):
                current_machine = solution.machines[j]
                last_pos = len(current_machine.job_schedule)
                factor = current_machine.simulate_remove_insert(-1, i, last_pos, instance)
                
                if min_factor == None or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
                    taken_machine = j
    
            curr_machine = solution.machines[taken_machine]
            last_pos = len(curr_machine.job_schedule)
            curr_machine.job_schedule.append( pm.Job(taken_job, -1, -1) )
            curr_machine.last_job = taken_job
            curr_machine.compute_objective(instance, startIndex=last_pos)
        
        solution.fix_objective()
        self.notify_on_solution_found(solution)
        self.notify_on_complete()
      
        return self.solve_result