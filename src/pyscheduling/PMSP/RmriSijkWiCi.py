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

class Heuristics(pm_methods.Heuristics):
    @staticmethod
    def list_heuristic(instance: RmriSijkWiCi_Instance, rule=1, decreasing=False):
        solution = ParallelMachines.ParallelSolution(instance)
        for machine in solution.machines:
            machine.wiTi_cache = []
            
        if rule == 1: #Earliest release date
            remaining_jobs_list = [(i, instance.R[i]) for i in range(instance.n)]
        elif rule == 2: #Min processing time
            remaining_jobs_list = [(i, min(instance.P[i])) for i in range(instance.n)] 
        elif rule == 3: #Mean Processing 
            remaining_jobs_list = [(i, mean(instance.P[i])) for i in range(instance.n)]
        elif rule == 4: #Max processing time
            remaining_jobs_list = [(i, max(instance.P[i])) for i in range(instance.n)]      
        elif rule == 5: #Min Sij*
            min_setup = [min([min(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, min_setup[i][0]) for i in range(instance.n)]    
        #Suite dispatching rules based on setup     
        elif rule == 6: #Earlist release date + mean processing time
            remaining_jobs_list = [(i,instance.R[i] + mean(instance.P[i])) for i in range(instance.n)]
        elif rule == 7: #Earliest release date + mean(Pi) + mean(Sij*)
            setup_means = [mean(means_list) for means_list in [
                [mean(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [
                (i, instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif rule == 8: #Earliest release date + mean(Pi) + mean(Si*j)
            setup_means = [mean(means_list) for means_list in [
                [mean(s[:,i]) for s in np.array(instance.S)] for i in range(instance.n)]]
            remaining_jobs_list = [
                (i, instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif rule == 9: #Earliest release date + mean(Pi) + mean(Sij*) + mean(Si*j)
            setup_means = [mean(means_list) for means_list in 
                        [[mean(s[i]) + mean(s[:,i]) for s in np.array(instance.S)] for i in range(instance.n)]]
            remaining_jobs_list = [(i,instance.R[i] + mean(instance.P[i]) + setup_means[i]) for i in range(instance.n)]
           
        remaining_jobs_list = sorted(remaining_jobs_list,key=lambda x:x[1],reverse=decreasing)
        jobs_list = [element[0] for element in remaining_jobs_list]
        
        return Heuristics.ordered_constructive(instance, remaining_jobs_list=jobs_list)