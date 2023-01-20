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

@parallel_instance([Constraints.R,Constraints.D,Constraints.W,Constraints.S], Objective.wiTi)
class RmridiSijkWiTi_Instance(ParallelMachines.ParallelInstance):
    
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
            min_witi_j = None
            for k in M:
                for i in E:  # (i for i in E if i != j ):
                    cj = self.R[j] + self.P[j][k] + self.S[k][i][j]
                    witi_j = self.W[j]*max(0,cj - self.D[j])
                    if min_witi_j is None or witi_j < min_witi_j:
                        min_witi_j = witi_j
            LB += min_witi_j
    
        return LB

class Heuristics(pm_methods.Heuristics):
    @staticmethod
    def list_heuristic(instance: RmridiSijkWiTi_Instance, rule=1, decreasing=False):
        """contains a list of static dispatching rules to be chosen from

        Args:
            instance (RmridiSijkWiTi_Instance): Instance to be solved
            rule_number (int, optional) : Index of the rule to use. Defaults to 1.

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        start_time = perf_counter()
        solution = ParallelMachines.ParallelSolution(instance)
        
        for machine in solution.machines:
            machine.wiTi_cache = []
            
        if rule == 1: #Earliest due dates 
            remaining_jobs_list = [(i, instance.D[i])
                                   for i in range(instance.n)]
        elif rule == 2: #Earliest release dates
            remaining_jobs_list = [(i,instance.R[i]) for i in range(instance.n)]    # type: ignore
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
        jobs_list = [element[0] for element in remaining_jobs_list]
        
        return Heuristics.ordered_constructive(instance, remaining_jobs_list=jobs_list)

class Metaheuristics(pm_methods.Metaheuristics):
    pass