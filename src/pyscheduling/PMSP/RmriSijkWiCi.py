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
from pyscheduling.Problem import Job

@parallel_instance([Constraints.R,Constraints.W,Constraints.S], Objective.wiCi)
class RmriSijkWiCi_Instance(ParallelMachines.ParallelInstance):
    
    def init_sol_method(self):
        return Heuristics.BIBA
    
    def lower_bound(self):
        """Computes the lower bound of sum(WiTi) of the instance 
        from the minimal completion time between job pairs on the number of machines

        Returns:
            int: Lower Bound of sum(WiTi)
        """
        # Preparing ranges
        M = range(self.m)
        E = range(self.n)
        # Compute lower bound
        LB = 0
        for j in E:
            min_wici_j = None
            for k in M:
                for i in E:  # (i for i in E if i != j ):
                    cj = self.R[j] + self.P[j][k] + self.S[k][i][j]
                    wici_j = self.W[j]*cj
                    if min_wici_j is None or wici_j < min_wici_j:
                        min_wici_j = wici_j
            LB += min_wici_j
    
        return LB

class Heuristics(pm_methods.Heuristics):
    @staticmethod
    def list_heuristic(instance: RmriSijkWiCi_Instance, rule=1, decreasing=False):
        """contains a list of static dispatching rules to be chosen from

        Args:
            instance (RmriSijkWiCi_Instance): Instance to be solved
            rule_number (int, optional) : Index of the rule to use. Defaults to 1.

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        solution = ParallelMachines.ParallelSolution(instance)
        for machine in solution.machines:
            machine.wiCi_cache = []
  
        if rule == 1: #R
            remaining_jobs_list = [(i, instance.R[i]) for i in range(instance.n)]
        elif rule == 2: #Min P
            remaining_jobs_list = [(i, min(instance.P[i])) for i in range(instance.n)] 
        elif rule == 3: #Mean P
            remaining_jobs_list = [(i, mean(instance.P[i])) for i in range(instance.n)]
        elif rule == 4: #Max P
            remaining_jobs_list = [(i, max(instance.P[i])) for i in range(instance.n)]      
        elif rule == 5: #Min Sij*
            setup_mins = [
                min(min_list) for min_list in [[min(s[i]) for s in instance.S]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_mins[i])
                                   for i in range(instance.n)]
        elif rule == 6: # Mean Sij*
            setup_means = [
                mean(means_list)
                for means_list in [[mean(s[i]) for s in instance.S]
                                   for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_means[i])
                                   for i in range(instance.n)]
        elif rule == 7: # Max Sij*
            setup_max = [
                max(max_list) for max_list in [[max(s[i]) for s in instance.S]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_max[i])
                                   for i in range(instance.n)]
        elif rule == 8: #Min Si*k
            setup_mins = [
                min(min_list) for min_list in [[min(s[:,i]) for s in np.array(instance.S)]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_mins[i])
                                   for i in range(instance.n)]    
        elif rule == 9: # Mean Si*k
            setup_means = [
                mean(means_list)
                for means_list in [[mean(s[:,i]) for s in np.array(instance.S)]
                                   for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_means[i])
                                   for i in range(instance.n)]
        elif rule == 10: # Max Si*k
            setup_max = [
                max(max_list) for max_list in [[max(s[:,i]) for s in np.array(instance.S)]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_max[i])
                                   for i in range(instance.n)]
        elif rule == 11: #R + Min P
            remaining_jobs_list = [(i,instance.R[i] + min(instance.P[i])) for i in range(instance.n)]
        elif rule == 12:
            remaining_jobs_list = [(i,instance.R[i] + mean(instance.P[i])) for i in range(instance.n)]
        elif rule == 13:
            remaining_jobs_list = [(i,instance.R[i] + max(instance.P[i])) for i in range(instance.n)]
        elif rule == 14: #R + Min Sij*
            setup_mins = [
                min(min_list) for min_list in [[min(s[i]) for s in instance.S]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, instance.R[i] + setup_mins[i])
                                   for i in range(instance.n)]
        elif rule == 15: # R + Mean Sij*
            setup_means = [
                mean(means_list)
                for means_list in [[mean(s[i]) for s in instance.S]
                                   for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, instance.R[i] + setup_means[i])
                                   for i in range(instance.n)]
        elif rule == 16: # R +Max Sij*
            setup_max = [
                max(max_list) for max_list in [[max(s[i]) for s in instance.S]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i,instance.R[i] + setup_max[i])
                                   for i in range(instance.n)]
        elif rule == 17: #Min Si*k
            setup_mins = [
                min(min_list) for min_list in [[min(s[:,i]) for s in np.array(instance.S)]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, instance.R[i] + setup_mins[i])
                                   for i in range(instance.n)]    
        elif rule == 18: # Mean Si*k
            setup_means = [
                mean(means_list)
                for means_list in [[mean(s[:,i]) for s in np.array(instance.S)]
                                   for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i,instance.R[i] + setup_means[i])
                                   for i in range(instance.n)]
        elif rule == 19: # Max Si*k
            setup_max = [
                max(max_list) for max_list in [[max(s[:,i]) for s in np.array(instance.S)]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i,instance.R[i] + setup_max[i])
                                   for i in range(instance.n)]
        elif rule == 20: #R + mean P + mean Sij*
            setup_means = [mean(means_list) for means_list in [
                [mean(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [
                (i, instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif rule == 21: #R + mean P + mean Si*j
            setup_means = [mean(means_list) for means_list in [
                [mean(s[:,i]) for s in np.array(instance.S)] for i in range(instance.n)]]
            remaining_jobs_list = [
                (i, instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif rule == 22: #R + mean P + mean Sij* + mean Si*j
            setup_means = [mean(means_list) for means_list in 
                        [[mean(s[i]) + mean(s[:,i]) for s in np.array(instance.S)] for i in range(instance.n)]]
            remaining_jobs_list = [(i,instance.R[i] + mean(instance.P[i]) + setup_means[i]) for i in range(instance.n)]
           
        remaining_jobs_list = sorted(remaining_jobs_list,key=lambda x:x[1],reverse=decreasing)
        jobs_list = [element[0] for element in remaining_jobs_list]
        
        return Heuristics.ordered_constructive(instance, remaining_jobs_list=jobs_list)

class Metaheuristics(pm_methods.Metaheuristics):
    pass