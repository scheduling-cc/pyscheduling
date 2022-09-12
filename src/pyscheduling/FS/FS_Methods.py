import random
from time import perf_counter

import pyscheduling.Problem as RootProblem
import pyscheduling.FS.FlowShop as FlowShop

class Heuristics_Cmax():

    def MINIT(instance : FlowShop.FlowShopInstance):
        """Gupta's MINIT heuristic which is based on iteratively scheduling a new job at the end
        so that it minimizes the idle time at the last machine

        Args:
            instance (FlowShop.FlowShopInstance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        start_time = perf_counter()
        solution = FlowShop.FlowShopSolution(instance=instance)

        #step 1 : Find pairs of jobs (job_i,job_j) which minimizes the idle time
        min_idleTime = None
        idleTime_ij_list = []
        for job_i in range(instance.n):
            for job_j in range(instance.n):
                if job_i != job_j :
                    pair = (job_i,job_j)
                    solution.job_schedule = [job_i,job_j] 
                    solution.cmax()
                    idleTime_ij = solution.idle_time()
                    idleTime_ij_list.append((pair,idleTime_ij))
                    if min_idleTime is None or idleTime_ij<min_idleTime : min_idleTime = idleTime_ij

        min_IT_list = [pair_idleTime_couple for pair_idleTime_couple in idleTime_ij_list if pair_idleTime_couple[1] == min_idleTime]
        #step 2 : Break the tie by choosing the pair based on performance at increasingly earlier machines (m-2,m-3,..)
        # For simplicity purposes, a random choice is performed
        min_IT = random.choice(min_IT_list)
                
        taken_pair = min_IT[0]
        job_schedule = [taken_pair[0],taken_pair[1]]
        solution.job_schedule = job_schedule
        solution.cmax()
        #step 3 :
        remaining_jobs_list = [job_id for job_id in list(range(instance.n)) if job_id not in job_schedule]

        while len(remaining_jobs_list) > 0 :
            min_IT_factor = None
            old_idleTime = solution.idle_time()
            for job_id in remaining_jobs_list:
                last_job_startTime, new_cmax = solution.idle_time_cmax_insert_last_pos(job_id)
                factor = old_idleTime + (last_job_startTime - solution.machines[instance.m-1].objective)
                if min_IT_factor is None or factor < min_IT_factor:
                    min_IT_factor = factor
                    taken_job = job_id
            job_schedule.append(taken_job)
            remaining_jobs_list.remove(taken_job)
            solution.job_schedule = job_schedule
            solution.cmax(len(job_schedule)-1)


        return RootProblem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])