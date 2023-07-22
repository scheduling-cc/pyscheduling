from dataclasses import dataclass
from statistics import mean
from typing import ClassVar, List


import pyscheduling.PMSP.ParallelMachines as ParallelMachines
from pyscheduling.PMSP.ParallelMachines import Constraints
from pyscheduling.Problem import Objective
from pyscheduling.core.base_solvers import BaseSolver
from pyscheduling.PMSP.solvers import BIBA, OrderedConstructive


@dataclass(init=False)
class RmriSijkWiCi_Instance(ParallelMachines.ParallelInstance):
    
    P: List[List[int]]
    W: List[int]
    R: List[int]
    S: List[List[List[int]]]
    constraints: ClassVar[Constraints] = [Constraints.P, Constraints.W, Constraints.R, Constraints.S]
    objective: ClassVar[Objective] = Objective.wiCi
    init_sol_method: BaseSolver = BIBA()
    
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

@dataclass
class ListHeuristic(BaseSolver):

    rule: int = 1
    decreasing : bool = False

    def solve(self, instance: RmriSijkWiCi_Instance):
        """contains a list of static dispatching rules to be chosen from

        Args:
            instance (RmriSijkWiCi_Instance): Instance to be solved
            rule_number (int, optional) : Index of the rule to use. Defaults to 1.

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        N = range(instance.n)
        M = range(instance.m)
        
        if self.rule == 1: #R
            remaining_jobs_list = [(i, instance.R[i]) for i in range(instance.n)]
        elif self.rule == 2: #Min P
            remaining_jobs_list = [(i, min(instance.P[i])) for i in range(instance.n)] 
        elif self.rule == 3: #Mean P
            remaining_jobs_list = [(i, mean(instance.P[i])) for i in range(instance.n)]
        elif self.rule == 4: #Max P
            remaining_jobs_list = [(i, max(instance.P[i])) for i in range(instance.n)]      
        elif self.rule == 5: #Min Sij*
            setup_mins = [
                min(min_list) for min_list in [[min(s[i]) for s in instance.S]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_mins[i])
                                   for i in range(instance.n)]
        elif self.rule == 6: # Mean Sij*
            setup_means = [
                mean(means_list)
                for means_list in [[mean(s[i]) for s in instance.S]
                                   for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_means[i])
                                   for i in range(instance.n)]
        elif self.rule == 7: # Max Sij*
            setup_max = [
                max(max_list) for max_list in [[max(s[i]) for s in instance.S]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_max[i])
                                   for i in range(instance.n)]
        elif self.rule == 8: #Min Si*k
            setup_mins_per_M = [[min(instance.S[i][k][j] for k in N) for j in N] for i in M ]
            setup_mins = [ min(setup_mins_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [(i, setup_mins[i]) for i in range(instance.n)]    
        elif self.rule == 9: # Mean Si*k
            setup_means_per_M = [[mean(instance.S[i][j][k] for k in N) for j in N] for i in M ]
            setup_means = [ mean(setup_means_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [(i, setup_means[i])
                                   for i in range(instance.n)]
        elif self.rule == 10: # Max Si*k
            setup_max_per_M = [[max(instance.S[i][k][j] for k in N) for j in N] for i in M ]
            setup_max = [ max(setup_max_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [(i, setup_max[i]) for i in range(instance.n)]
        elif self.rule == 11: #R + Min P
            remaining_jobs_list = [(i,instance.R[i] + min(instance.P[i])) for i in range(instance.n)]
        elif self.rule == 12:
            remaining_jobs_list = [(i,instance.R[i] + mean(instance.P[i])) for i in range(instance.n)]
        elif self.rule == 13:
            remaining_jobs_list = [(i,instance.R[i] + max(instance.P[i])) for i in range(instance.n)]
        elif self.rule == 14: #R + Min Sij*
            setup_mins = [
                min(min_list) for min_list in [[min(s[i]) for s in instance.S]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, instance.R[i] + setup_mins[i])
                                   for i in range(instance.n)]
        elif self.rule == 15: # R + Mean Sij*
            setup_means = [
                mean(means_list)
                for means_list in [[mean(s[i]) for s in instance.S]
                                   for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, instance.R[i] + setup_means[i])
                                   for i in range(instance.n)]
        elif self.rule == 16: # R +Max Sij*
            setup_max = [
                max(max_list) for max_list in [[max(s[i]) for s in instance.S]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i,instance.R[i] + setup_max[i])
                                   for i in range(instance.n)]
        elif self.rule == 17: #Min Si*k
            setup_mins_per_M = [[min(instance.S[i][k][j] for k in N) for j in N] for i in M ]
            setup_mins = [ min(setup_mins_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [(i, instance.R[i] + setup_mins[i])
                                   for i in range(instance.n)]    
        elif self.rule == 18: # Mean Si*k
            setup_means_per_M = [[mean(instance.S[i][j][k] for k in N) for j in N] for i in M ]
            setup_means = [ mean(setup_means_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [(i,instance.R[i] + setup_means[i])
                                   for i in range(instance.n)]
        elif self.rule == 19: # Max Si*k
            setup_max_per_M = [[max(instance.S[i][k][j] for k in N) for j in N] for i in M ]
            setup_max = [ max(setup_max_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [(i,instance.R[i] + setup_max[i])
                                   for i in range(instance.n)]
        elif self.rule == 20: #R + mean P + mean Sij*
            setup_means = [mean(means_list) for means_list in [
                [mean(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [
                (i, instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif self.rule == 21: #R + mean P + mean Si*j
            setup_means_per_M = [[mean(instance.S[i][j][k] for k in N) for j in N] for i in M ]
            setup_means = [ mean(setup_means_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [
                (i, instance.R[i] + mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif self.rule == 22: #R + mean P + mean Sij* + mean Si*j
            setup_means_per_M = [[mean(instance.S[i][j][k] for k in N) + mean(instance.S[i][k][j] for k in N) for j in N] for i in M ]
            setup_means = [ mean(setup_means_per_M[i][j] for i in M) for j in N ]

            remaining_jobs_list = [(i,instance.R[i] + mean(instance.P[i]) + setup_means[i]) for i in range(instance.n)]
           
        remaining_jobs_list = sorted(remaining_jobs_list,
                                     key = lambda x:x[1],
                                     reverse = self.decreasing)
        jobs_list = [element[0] for element in remaining_jobs_list]
        
        return OrderedConstructive(remaining_jobs_list=jobs_list).solve(instance)
