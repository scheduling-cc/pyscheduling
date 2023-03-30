from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import ClassVar, List

import pyscheduling.PMSP.ParallelMachines as ParallelMachines
import pyscheduling.PMSP.PM_methods as pm_methods
import pyscheduling.Problem as Problem
from pyscheduling.PMSP.ParallelMachines import Constraints
from pyscheduling.Problem import Job, Objective, Solver


@dataclass(init=False)
class RmridiSijkWiTi_Instance(ParallelMachines.ParallelInstance):
    
    P: List[List[int]]
    W: List[int]
    R: List[int]
    D: List[int]
    S: List[List[List[int]]]
    constraints: ClassVar[Constraints] = [Constraints.P, Constraints.W, Constraints.R, Constraints.D, Constraints.S]
    objective: ClassVar[Objective] = Objective.wiTi

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
        
        N = range(instance.n)
        M = range(instance.m)

        for machine in solution.machines:
            machine.wiTi_cache = []
            
        if rule == 1: #Earliest due dates 
            remaining_jobs_list = [(i, instance.D[i])
                                   for i in N]
        elif rule == 2: #Earliest release dates
            remaining_jobs_list = [(i,instance.R[i]) for i in N]    # type: ignore
        elif rule == 3: #Earlist due date + mean processing time
            remaining_jobs_list = [(i,instance.D[i] + mean(instance.P[i])) for i in N]
        elif rule == 4:#Earlist release date + mean processing time
            remaining_jobs_list = [(i,instance.R[i] + mean(instance.P[i])) for i in N]
        elif rule == 5:  #min(due date - release date) 
            remaining_jobs_list = [(i,instance.D[i] - instance.R[i]) for i in N]
        elif rule == 6: #release date + mean(Pi) + mean(Sij*)
            setup_means_per_M = [[mean(instance.S[i][k][j] for k in N) for j in N] for i in M ]
            setup_means = [ mean(setup_means_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [
                (i, instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in N]
        elif rule == 7: #release date + mean(Pi) + mean(Si*j)
            setup_means_per_M = [[mean(instance.S[i][j][k] for k in N) for j in N] for i in M ]
            setup_means = [ mean(setup_means_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [
                (i, instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in N]
        elif rule == 8: #release date + mean(Pi) + mean(Sij*) + mean(Si*j)
            setup_means_per_M = [[mean(instance.S[i][j][k] for k in N) + mean(instance.S[i][k][j] for k in N) for j in N] for i in M ]
            setup_means = [ mean(setup_means_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [(i,instance.R[i] + mean(instance.P[i]) + setup_means[i]) for i in N]
        elif rule == 9:#min(due date - release date) + mean(Pi)
            remaining_jobs_list = [(i,instance.D[i] - instance.R[i] + mean(instance.P[i])) for i in N]
        elif rule == 10:#min(due date - release date) + mean(Pi) + mean(Sij*)
            setup_means = [mean(means_list) for means_list in [
                [mean(s[i]) for s in instance.S] for i in N]]
            remaining_jobs_list = [
                (i, instance.D[i] - instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in N]
        elif rule == 11: #min(due date - release date) + mean(Pi) + mean(Si*j)
            setup_means_per_M = [[mean(instance.S[i][k][j] for k in N) for j in N] for i in M ]
            setup_means = [ mean(setup_means_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [
                (i, instance.D[i] - instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in N]
        elif rule == 12: #min(due date - release date) + mean(Pi) + mean(Sij*) + mean(Si*j)
            setup_means_per_M = [[mean(instance.S[i][j][k] for k in N) + mean(instance.S[i][k][j] for k in N) for j in N] for i in M ]
            setup_means = [ mean(setup_means_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [(i,instance.D[i] - instance.R[i] + mean(instance.P[i]) + setup_means[i]) for i in N]
              
        remaining_jobs_list = sorted(remaining_jobs_list,key=lambda x:x[1],reverse=decreasing)
        jobs_list = [element[0] for element in remaining_jobs_list]
        
        return Heuristics.ordered_constructive(instance, remaining_jobs_list=jobs_list)

class Metaheuristics(pm_methods.Metaheuristics):
    pass