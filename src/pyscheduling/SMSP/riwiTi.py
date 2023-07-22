from dataclasses import dataclass
from math import exp
from typing import ClassVar, List

import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.core.base_solvers import BaseSolver
from pyscheduling.Problem import Objective
from pyscheduling.SMSP.SingleMachine import Constraints

@dataclass(init=False)
class riwiTi_Instance(SingleMachine.SingleInstance):

    P: List[int]
    W: List[int]
    R: List[int]
    D: List[int]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.W, Constraints.R, Constraints.D]
    objective: ClassVar[Objective] = Objective.wiTi

    @property
    def init_sol_method(self):
        return ACT_WSECi()


class ACT_WSECi(BaseSolver):

    def solve(self, instance : riwiTi_Instance):
        """Appearant Tardiness Cost heuristic using WSECi rule instead of WSPT.

        Args:
            instance (riwiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: Solve Result of the instance by the method
        """
        self.notify_on_start()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiTi_cache = []
        ci = 0
        wiTi = 0
        remaining_jobs_list = list(range(instance.n))
        sumP = sum(instance.P)
        K = self.ACT_Tuning(instance)
        rule = lambda job_id : (float(instance.W[job_id])/float(max(instance.R[job_id] - ci,0) + instance.P[job_id]))*exp(-max(instance.D[job_id]-instance.P[job_id]-ci,0)/(K*sumP))
        while(len(remaining_jobs_list)>0):
            remaining_jobs_list.sort(reverse=True,key=rule)
            taken_job = remaining_jobs_list[0]
            start_time = max(instance.R[taken_job],ci)
            ci = start_time + instance.P[taken_job]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,start_time,ci))
            wiTi += instance.W[taken_job]*max(ci-instance.D[taken_job],0)
            solution.machine.wiTi_cache.append(wiTi)
            remaining_jobs_list.pop(0)
        solution.machine.objective_value=solution.machine.wiTi_cache[instance.n-1]
        solution.fix_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result 
    
    def ACT_Tuning(self, instance : riwiTi_Instance):
        """Analyze the instance to consequently tune the ACT. For now, the tuning is static.

        Args:
            instance (riwiTi_Instance): Instance tackled by ACT heuristic

        Returns:
            int, int: K
        """
        Tightness = 1 - sum(instance.D)/(instance.n*sum(instance.P))
        Range = (max(instance.D)-min(instance.D))/sum(instance.P)
        return 0.2


@dataclass
class ACT_WSAPT(BaseSolver):

    def solve(self, instance : riwiTi_Instance):
        """Appearant Tardiness Cost heuristic using WSAPT rule instead of WSPT

        Args:
            instance (riwiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: Solve Result of the instance by the method
        """
        self.notify_on_start()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiTi_cache = []
        ci = min(instance.R)
        wiTi = 0
        remaining_jobs_list = list(range(instance.n))
        sumP = sum(instance.P)
        K = self.ACT_Tuning(instance)
        rule = lambda job_id : (float(instance.W[job_id])/float(instance.P[job_id]))*exp(-max(instance.D[job_id]-instance.P[job_id]-ci,0)/(K*sumP))
        while(len(remaining_jobs_list)>0):
            filtered_remaining_jobs_list = list(filter(lambda job_id : instance.R[job_id]<=ci,remaining_jobs_list))
            filtered_remaining_jobs_list.sort(reverse=True,key=rule)
            if(len(filtered_remaining_jobs_list)==0):
                ci = min([instance.R[job_id] for job_id in remaining_jobs_list])
                filtered_remaining_jobs_list = list(filter(lambda job_id : instance.R[job_id]<=ci,remaining_jobs_list))
                filtered_remaining_jobs_list.sort(reverse=True,key=rule)
            taken_job = remaining_jobs_list[0]
            start_time = max(instance.R[taken_job],ci)
            ci = start_time + instance.P[taken_job]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,start_time,ci))
            wiTi += instance.W[taken_job]*max(ci-instance.D[taken_job],0)
            solution.machine.wiTi_cache.append(wiTi)
            remaining_jobs_list.pop(0)
        solution.machine.objective_value=solution.machine.wiTi_cache[instance.n-1]
        solution.fix_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result 
    
    def ACT_Tuning(self, instance : riwiTi_Instance):
        """Analyze the instance to consequently tune the ACT. For now, the tuning is static.

        Args:
            instance (riwiTi_Instance): Instance tackled by ACT heuristic

        Returns:
            int, int: K
        """
        Tightness = 1 - sum(instance.D)/(instance.n*sum(instance.P))
        Range = (max(instance.D)-min(instance.D))/sum(instance.P)
        return 0.2
    