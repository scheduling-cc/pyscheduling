from dataclasses import dataclass
from math import exp
from typing import ClassVar, List

import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.core.base_solvers import BaseSolver
from pyscheduling.Problem import Objective
from pyscheduling.SMSP.SingleMachine import Constraints


@dataclass(init=False)
class risijwiTi_Instance(SingleMachine.SingleInstance):

    P: List[int]
    W: List[int]
    R: List[int]
    D: List[int]
    S: List[List[int]]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.W, Constraints.R, Constraints.D, Constraints.S]
    objective: ClassVar[Objective] = Objective.wiTi
    
    @property
    def init_sol_method(self):
        return ACTS_WSECi()


@dataclass
class ACTS_WSECi(BaseSolver):
    
    def solve(self, instance : risijwiTi_Instance):
        """Appearant Tardiness Cost with Setup heuristic using WSECi rule instead of WSPT

        Args:
            instance (risijwiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: Solve Result of the instance by the method
        """
        self.notify_on_start()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiTi_cache = []
        ci = 0
        wiTi = 0
        prev_job = -1
        remaining_jobs_list = list(range(instance.n))
        while(len(remaining_jobs_list)>0):
            prev_job, taken_job = self.ACTS_WSECi_Sorting(instance,remaining_jobs_list,ci,prev_job)
            start_time = max(instance.R[taken_job],ci)
            ci = start_time + instance.S[prev_job][taken_job] + instance.P[taken_job]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,start_time,ci))
            wiTi += instance.W[taken_job]*max(ci-instance.D[taken_job],0)
            solution.machine.wiTi_cache.append(wiTi)
            remaining_jobs_list.remove(taken_job)
            prev_job = taken_job
        solution.machine.objective_value=solution.machine.wiTi_cache[instance.n-1]
        solution.fix_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result

    def ACTS_WSECi_Sorting(self, instance : risijwiTi_Instance, remaining_jobs : List[SingleMachine.Job], t : int, prev_job : int):
        """Returns the prev_job and the job to be scheduled next based on ACTS_WSECi rule.
        It returns a couple of previous job scheduled and the new job to be scheduled. The previous job will be the
        same than the taken job if it's the first time when the rule is applied, is the same prev_job passed as
        argument to the function otherwise. This is to avoid extra-ifs and thus not slowing the execution of 
        the heuristic

        Args:
            instance (risijwiTi_Instance): Instance tackled by the ACTS_WSECi heuristic
            remaining_jobs (list[SingleMachine.Job]): Remaining jobs list to be scheduled
            t (int): current time
            prev_job (int): Previous scheduled job, necessary for setup time

        Returns:
           int, int: previous job scheduled, taken job to be scheduled
        """
        sumP = sum(instance.P)
        sumS = 0
        for i in range(instance.n):
            sumSi = sum(instance.S[i])
            sumS += sumSi
        K1, K2 = self.ACTS_WSECi_Tuning(instance)
        rule = lambda prev_j,job_id : (float(instance.W[job_id])/float(max(instance.R[job_id] - t,0) + instance.P[job_id]))*exp(
            -max(instance.D[job_id]-instance.P[job_id]-t,0)/(K1*sumP))*exp(-instance.S[prev_j][job_id]/(K2*sumS))
        max_rule_value = -1
        if prev_job == -1:
            for job in remaining_jobs:
                rule_value = rule(job,job)
                if max_rule_value<rule_value: 
                    max_rule_value = rule_value
                    taken_job = job
            return taken_job, taken_job
        else:
            for job in remaining_jobs:
                rule_value = rule(prev_job,job)
                if max_rule_value<rule_value: 
                    max_rule_value = rule_value
                    taken_job = job
            return prev_job, taken_job
        

    def ACTS_WSECi_Tuning(self, instance : risijwiTi_Instance):
        """Analyze the instance to consequently tune the ACTS_WSECi. For now, the tuning is static.

        Args:
            instance (risijwiTi_Instance): Instance tackled by ACTS_WSECi heuristic

        Returns:
            int, int: K1 , K2
        """
        Tightness = 1 - sum(instance.D)/(instance.n*sum(instance.P))
        Range = (max(instance.D)-min(instance.D))/sum(instance.P)
        return 0.2, 1