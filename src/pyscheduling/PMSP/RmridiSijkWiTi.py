import heapq
import imp
from os import stat
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from random import randint, uniform
from statistics import mean
from time import perf_counter
from unittest import result


import matplotlib.pyplot as plt
import numpy as np 

import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Constraints, Objective, Solver
import pyscheduling.PMSP.ParallelMachines as ParallelMachines
from pyscheduling.PMSP.ParallelMachines import parallel_instance
import pyscheduling.PMSP.PM_methods as pm_methods

@parallel_instance([Constraints.R,Constraints.D,Constraints.W,Constraints.S], Objective.wiTi)
class RmridiSijkWiTi_Instance(ParallelMachines.ParallelInstance):
    
    def init_sol_method(self):
        return 
    
    def lower_bound(self):
        """Computes the lower bound of maximal completion time of the instance 
        by dividing the sum of minimal completion time between job pairs on the number of machines

        Returns:
            int: Lower Bound of maximal completion time
        """
        # Preparing ranges
        M = range(self.m)
        E = range(self.n)
        # Compute lower bound
        sum_j = 0
        all_max_r_j = 0
        for j in E:
            min_j = None
            min_r_j = None
            for k in M:
                for i in E:  # (t for t in E if t != j ):
                    if min_j is None or self.P[j][k] + self.S[k][i][j] < min_j:
                        min_j = self.P[j][k] + self.S[k][i][j]
                    if min_r_j is None or self.P[j][k] + self.S[k][i][j] < min_r_j:
                        min_r_j = self.P[j][k] + self.S[k][i][j]
            sum_j += min_j
            all_max_r_j = max(all_max_r_j, min_r_j)

        lb1 = sum_j / self.m
        LB = max(lb1, all_max_r_j)

        return LB

class Heuristics:
    @staticmethod
    def list_heuristic(instance: RmridiSijkWiTi_Instance, rule=1, decreasing=False):
        start_time = perf_counter()
        solution = ParallelMachines.ParallelSolution(instance=instance)
        
        if rule == 1: #Earliest due dates 
            remaining_jobs_list = [(i, instance.D[i])
                                   for i in range(instance.n)]
        elif rule == 2: #Earliest release dates
            remaining_jobs_list = [(i,instance.R[i]) for i in range(instance.n)]  
        elif rule == 3: #Earlist due date + mean processing time
            remaining_jobs_list = [(i,instance.D[i] + mean(instance.P[i])) for i in range(instance.n)]
        elif rule == 4:#Earlist release date + mean processing time
            remaining_jobs_list = [(i,instance.R[i] + mean(instance.P[i])) for i in range(instance.n)]
        elif rule == 5:  #min(due date - release date) 
            remaining_jobs_list = [(i,instance.D[i] - instance.R[i]) for i in range(instance.n)]
        elif rule == 6: #release date + mean(Pi) + mean(Sij*)
            setup_means = [mean(means_list) for means_list in [
                [mean(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [
                (i, instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif rule == 7: #release date + mean(Pi) + mean(Si*j)
            setup_means = [mean(means_list) for means_list in [
                [mean(s[:,i]) for s in np.array(instance.S)] for i in range(instance.n)]]
            remaining_jobs_list = [
                (i, instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif rule == 8: #release date + mean(Pi) + mean(Sij*) + mean(Si*j)
            setup_means = [mean(means_list) for means_list in 
                        [[mean(s[i]) + mean(s[:,i]) for s in np.array(instance.S)] for i in range(instance.n)]]
            remaining_jobs_list = [(i,instance.R[i] + mean(instance.P[i]) + setup_means[i]) for i in range(instance.n)]
        elif rule == 9:#min(due date - release date) + mean(Pi)
            remaining_jobs_list = [(i,instance.D[i] - instance.R[i] + mean(instance.P[i])) for i in range(instance.n)]
        elif rule == 10:#min(due date - release date) + mean(Pi) + mean(Sij*)
            setup_means = [mean(means_list) for means_list in [
                [mean(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [
                (i, instance.D[i] - instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif rule == 11: #min(due date - release date) + mean(Pi) + mean(Si*j)
            setup_means = [mean(means_list) for means_list in [
                [mean(s[:,i]) for s in np.array(instance.S)] for i in range(instance.n)]]
            remaining_jobs_list = [
                (i, instance.D[i] - instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif rule == 12: #min(due date - release date) + mean(Pi) + mean(Sij*) + mean(Si*j)
            setup_means = [mean(means_list) for means_list in 
                        [[mean(s[i]) + mean(s[:,i]) for s in np.array(instance.S)] for i in range(instance.n)]]
            remaining_jobs_list = [(i,instance.D[i] - instance.R[i] + mean(instance.P[i]) + setup_means[i]) for i in range(instance.n)]
              
        remaining_jobs_list = sorted(remaining_jobs_list,key=lambda x:x[1],reverse=decreasing)
        for element in remaining_jobs_list:
            i = element[0]
            min_wiTi = None
            start_time = None
            for j in range(instance.m):
                current_machine_schedule = solution.machines[j]
                wiTi = current_machine_schedule.weighted_lateness_insert(i,len(current_machine_schedule.job_schedule),instance) 
                if (min_wiTi == None) or (wiTi < min_wiTi):
                    taken_machine = j
                    min_wiTi = wiTi
                    release_time = max(instance.R[i] - current_machine_schedule.completion_time,0)
                    start_time = current_machine_schedule.completion_time + release_time
            # Apply the move
            if (solution.machines[taken_machine].last_job == -1):
                ci = start_time + instance.P[i][taken_machine] +\
                    instance.S[taken_machine][i][i]  # Added Sj_ii for rabadi
            else:
                ci = start_time + instance.P[i][
                    taken_machine] + instance.S[taken_machine][
                        solution.machines[taken_machine].last_job][i]
                            
            solution.machines[taken_machine].job_schedule.append(ParallelMachines.Job(
                i, start_time, ci))
            solution.machines[taken_machine].completion_time = ci  # type: ignore
            solution.machines[taken_machine].last_job = i
            solution.machines[taken_machine].objective = min_wiTi  # type: ignore
            solution.machines[taken_machine].wiTi_index.append(min_wiTi)
        
        solution.compute_objective(Objective.wiTi)
        #Add fix objective method according to the obj
        
        return RootProblem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])