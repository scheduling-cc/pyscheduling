from dataclasses import dataclass
from math import exp
from typing import ClassVar, List

import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.core.base_solvers import BaseSolver
from pyscheduling.Problem import Objective
from pyscheduling.SMSP.SingleMachine import Constraints


@dataclass(init=False)
class wiTi_Instance(SingleMachine.SingleInstance):

    P: List[int]
    W: List[int]
    D: List[int]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.W, Constraints.D]
    objective: ClassVar[Objective] = Objective.wiTi

    @property
    def init_sol_method(self):
        return ACT()


class WSPT(BaseSolver):

    def solve(self, instance : wiTi_Instance):
        """WSPT rule is efficient if the due dates are too tight (for overdue jobs)

        Args:
            instance (wiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
        jobs = list(range(instance.n))
        jobs.sort(reverse=True,key=lambda job_id : float(instance.W[job_id])/float(instance.P[job_id]))
        solution = SingleMachine.SingleSolution(instance)
        for job in jobs:
            solution.machine.job_schedule.append(SingleMachine.Job(job,0,0)) 
        solution.compute_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result 
    
class MS(BaseSolver):

    def solve(self, instance : wiTi_Instance):
        """MS rule is efficient if the due dates are too loose (for not overdue jobs)

        Args:
            instance (wiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiTi_cache = []
        ci = 0
        wiTi = 0
        remaining_jobs_list = list(range(instance.n))
        rule = lambda job_id : max(instance.D[job_id]-instance.P[job_id]-ci,0)
        while(len(remaining_jobs_list)>0):
            remaining_jobs_list.sort(reverse=False,key=rule)
            taken_job = remaining_jobs_list[0]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,ci,ci+instance.P[taken_job]))
            ci += instance.P[taken_job]
            wiTi += instance.W[taken_job]*max(ci-instance.D[taken_job],0)
            solution.machine.wiTi_cache.append(wiTi)
            remaining_jobs_list.pop(0)
        solution.machine.objective_value=solution.machine.wiTi_cache[instance.n-1]
        solution.fix_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result 
    
class ACT(BaseSolver):

    def solve(self, instance : wiTi_Instance):
        """Appearant Cost Tardiness rule balances between WSPT and MS rules based on due dates tightness and range

        Args:
            instance (wiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiTi_cache = []
        ci = 0
        wiTi = 0
        remaining_jobs_list = list(range(instance.n))
        sumP = sum(instance.P)
        K = self.ACT_Tuning(instance)
        rule = lambda job_id : (float(instance.W[job_id])/float(instance.P[job_id]))*exp(-max(instance.D[job_id]-instance.P[job_id]-ci,0)/(K*sumP))
        while(len(remaining_jobs_list)>0):
            remaining_jobs_list.sort(reverse=True,key=rule)
            taken_job = remaining_jobs_list[0]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,ci,ci+instance.P[taken_job]))
            ci += instance.P[taken_job]
            wiTi += instance.W[taken_job]*max(ci-instance.D[taken_job],0)
            solution.machine.wiTi_cache.append(wiTi)
            remaining_jobs_list.pop(0)
        solution.machine.objective_value=solution.machine.wiTi_cache[instance.n-1]
        solution.fix_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result 

    def ACT_Tuning(self, instance : wiTi_Instance):
        """Analyze the instance to consequently tune the ACT. For now, the tuning is static.

        Args:
            instance (riwiTi_Instance): Instance tackled by ACT heuristic

        Returns:
            int, int: K
        """
        Tightness = 1 - sum(instance.D)/(instance.n*sum(instance.P))
        Range = (max(instance.D)-min(instance.D))/sum(instance.P)
        return 0.2
    