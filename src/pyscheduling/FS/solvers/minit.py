from dataclasses import dataclass
import random

import pyscheduling.FS.FlowShop as FS
from pyscheduling import Problem
from pyscheduling.Problem import Job
from pyscheduling.core.base_solvers import BaseSolver


@dataclass
class MINIT(BaseSolver):

    def solve(self, instance : FS.FlowShopInstance):
        """Gupta's MINIT heuristic which is based on iteratively scheduling a new job at the end
        so that it minimizes the idle time at the last machine

        Args:
            instance (FlowShop.FlowShopInstance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
        solution = FS.FlowShopSolution(instance=instance)

        #step 1 : Find pairs of jobs (job_i,job_j) which minimizes the idle time
        min_idleTime = None
        idleTime_ij_list = []
        for job_i in range(instance.n):
            for job_j in range(instance.n):
                if job_i != job_j :
                    pair = (job_i,job_j)
                    solution.job_schedule = [Job(job_i, 0, 0),Job(job_j, 0, 0)] 
                    solution.compute_objective()
                    idleTime_ij = solution.idle_time()
                    idleTime_ij_list.append((pair,idleTime_ij))
                    if min_idleTime is None or idleTime_ij<min_idleTime : min_idleTime = idleTime_ij

        min_IT_list = [pair_idleTime_couple for pair_idleTime_couple in idleTime_ij_list if pair_idleTime_couple[1] == min_idleTime]
        #step 2 : Break the tie by choosing the pair based on performance at increasingly earlier machines (m-2,m-3,..)
        # For simplicity purposes, a random choice is performed
        min_IT = random.choice(min_IT_list)
        
        i, j = min_IT[0] # Taken pair
        job_schedule = [Job(i, 0, 0), Job(j, 0, 0)]
        solution.job_schedule = job_schedule
        solution.compute_objective()
        #step 3 :
        remaining_jobs_list = [job_id for job_id in list(range(instance.n)) if job_id not in {i, j}]

        while len(remaining_jobs_list) > 0 :
            min_IT_factor = None
            old_idleTime = solution.idle_time()
            for job_id in remaining_jobs_list:
                last_job_startTime, new_cmax = solution.simulate_insert_last(job_id)
                factor = old_idleTime + (last_job_startTime - solution.machines[instance.m-1].objective_value)
                if min_IT_factor is None or factor < min_IT_factor:
                    min_IT_factor = factor
                    taken_job = job_id

            solution.job_schedule.append(Job(taken_job, 0, 0))
            remaining_jobs_list.remove(taken_job)
            solution.compute_objective(startIndex=len(job_schedule)-1)

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result
