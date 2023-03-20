from math import exp
from time import perf_counter

import pyscheduling.Problem as RootProblem
import pyscheduling.SMSP.SingleMachine as SingleMachine
import pyscheduling.SMSP.SM_methods as Methods
from pyscheduling.Problem import Constraints, Objective
from pyscheduling.SMSP.SingleMachine import single_instance
from pyscheduling.SMSP.SM_methods import ExactSolvers


@single_instance([Constraints.W, Constraints.R, Constraints.D], Objective.wiTi)
class riwiTi_Instance(SingleMachine.SingleInstance):

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.ACT_WSECi


class Heuristics():

    @staticmethod
    def ACT_WSECi(instance : riwiTi_Instance):
        """Appearant Tardiness Cost heuristic using WSECi rule instead of WSPT.

        Args:
            instance (riwiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: Solve Result of the instance by the method
        """
        startTime = perf_counter()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiTi_cache = []
        ci = 0
        wiTi = 0
        remaining_jobs_list = list(range(instance.n))
        sumP = sum(instance.P)
        K = Heuristics_Tuning.ACT(instance)
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
        return RootProblem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])
    
    @staticmethod
    def ACT_WSAPT(instance : riwiTi_Instance):
        """Appearant Tardiness Cost heuristic using WSAPT rule instead of WSPT

        Args:
            instance (riwiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: Solve Result of the instance by the method
        """
        startTime = perf_counter()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiTi_cache = []
        ci = min(instance.R)
        wiTi = 0
        remaining_jobs_list = list(range(instance.n))
        sumP = sum(instance.P)
        K = Heuristics_Tuning.ACT(instance)
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
        return RootProblem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])
    
    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]


class Metaheuristics(Methods.Metaheuristics):
    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

class Heuristics_Tuning():

    @staticmethod
    def ACT(instance : riwiTi_Instance):
        """Analyze the instance to consequently tune the ACT. For now, the tuning is static.

        Args:
            instance (riwiTi_Instance): Instance tackled by ACT heuristic

        Returns:
            int, int: K
        """
        Tightness = 1 - sum(instance.D)/(instance.n*sum(instance.P))
        Range = (max(instance.D)-min(instance.D))/sum(instance.P)
        return 0.2