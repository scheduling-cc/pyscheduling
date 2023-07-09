from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, ClassVar, List

import pyscheduling.Problem as Problem
import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.Problem import Objective
from pyscheduling.SMSP.SingleMachine import Constraints
from pyscheduling.core.base_solvers import BaseSolver


@dataclass(init=False)
class riwiCi_Instance(SingleMachine.SingleInstance):
    
    P: List[int]
    W: List[int]
    R: List[int]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.W, Constraints.R]
    objective: ClassVar[Objective] = Objective.wiCi

    @property
    def init_sol_method(self):
        return WSECi()


@dataclass
class WSECi(BaseSolver):

    def solve(self, instance : riwiCi_Instance):
        """Weighted Shortest Expected Completion time, dynamic dispatching rule inspired from WSPT but adds release
        time to processing time

        Args:
            instance (riwiCi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiCi_cache = []
        ci = 0
        wiCi = 0
        remaining_jobs_list = list(range(instance.n))
        rule = lambda job_id : float(instance.W[job_id])/float(max(instance.R[job_id] - ci,0) + instance.P[job_id])
        while(len(remaining_jobs_list)>0):
            remaining_jobs_list.sort(reverse=True,key=rule)
            taken_job = remaining_jobs_list[0]
            start_time = max(instance.R[taken_job],ci)
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,start_time,start_time+instance.P[taken_job]))
            ci = start_time+instance.P[taken_job]
            wiCi += instance.W[taken_job]*ci
            solution.machine.wiCi_cache.append(wiCi)
            remaining_jobs_list.pop(0)
        solution.machine.objective_value=solution.machine.wiCi_cache[instance.n-1]
        solution.fix_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result 

@dataclass
class WSAPT(BaseSolver):

    def solve(self, instance : riwiCi_Instance):
        """Weighted Shortest Available Processing time, dynamic dispatching rule inspired from WSPT but considers
        available jobs only at a given time t

        Args:
            instance (riwiCi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiCi_cache = []
        ci = min(instance.R)
        wiCi = 0
        remaining_jobs_list = list(range(instance.n))
        
        rule = lambda job_id : float(instance.W[job_id])/float(instance.P[job_id])
        while(len(remaining_jobs_list)>0):
            filtered_remaining_jobs_list = list(filter(lambda job_id : instance.R[job_id]<=ci,remaining_jobs_list))
            filtered_remaining_jobs_list.sort(reverse=True,key=rule)
            if(len(filtered_remaining_jobs_list)==0):
                ci = min([instance.R[job_id] for job_id in remaining_jobs_list])
                filtered_remaining_jobs_list = list(filter(lambda job_id : instance.R[job_id]<=ci,remaining_jobs_list))
                filtered_remaining_jobs_list.sort(reverse=True,key=rule)

            taken_job = filtered_remaining_jobs_list[0]
            start_time = max(instance.R[taken_job],ci)
            ci = start_time+instance.P[taken_job]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,start_time,ci))
            wiCi += instance.W[taken_job]*ci
            solution.machine.wiCi_cache.append(wiCi)
            remaining_jobs_list.remove(taken_job)

        solution.machine.objective_value=solution.machine.wiCi_cache[instance.n-1]
        solution.fix_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result 
    

@dataclass
class ListHeuristic(BaseSolver):

    rule_number : int = 1
    reverse : bool = False

    def solve(self, instance : riwiCi_Instance):
        """contains a list of static dispatching rules to be chosen from

        Args:
            instance (riwiCi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiCi_cache = []
        if self.rule_number==1: # Increasing order of the release time
            sorting_func = lambda job_id : instance.R[job_id]
        elif self.rule_number==2: # WSPT
            sorting_func = lambda job_id : float(instance.W[job_id])/float(instance.P[job_id])
        elif self.rule_number ==3: #WSPT including release time in the processing time
            sorting_func = lambda job_id : float(instance.W[job_id])/float(instance.R[job_id]+instance.P[job_id])

        remaining_jobs_list = list(range(instance.n))
        remaining_jobs_list.sort(reverse=self.reverse,key=sorting_func)
        
        ci = 0
        wiCi = 0
        for job in remaining_jobs_list:
            start_time = max(instance.R[job],ci)
            ci = start_time + instance.P[job]
            wiCi += instance.W[job]*ci
            solution.machine.job_schedule.append(SingleMachine.Job(job,start_time,ci))
            solution.machine.wiCi_cache.append(wiCi)
        solution.machine.objective_value = solution.machine.wiCi_cache[instance.n - 1]
        solution.fix_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result 
    
@dataclass
class ILS(BaseSolver):

    time_limit_factor: int = None
    init_sol_method: Callable = field(default_factory=WSECi)
    Nb_iter: int = 500000
    Non_improv: int = 5000

    def solve(self, instance : riwiCi_Instance, ** kwargs):
        """Applies LocalSearch on the current solution iteratively

        Args:
            instance (riwiCi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
        first_time = perf_counter()
        if self.time_limit_factor:
            time_limit = instance.n * self.time_limit_factor

        # Generate init solutoin using the initial solution method
        solution_init = self.init_sol_method.solve(instance).best_solution

        if not solution_init:
            return Problem.SolveResult()

        local_search = SingleMachine.SM_LocalSearch(copy_solution=True)

        all_solutions = []
        solution_best = solution_init.copy()  # Save the current best solution
        solution_i = solution_init.copy()
        all_solutions.append(solution_best)

        N = 0
        i = 0
        while i < self.Nb_iter and N < self.Non_improv:
            # check time limit if exists
            if self.time_limit_factor and (perf_counter() - first_time) >= time_limit:
                break

            solution_i = local_search.improve(solution_i)
            self.notify_on_solution_found(solution_i)

            if solution_i.objective_value < solution_best.objective_value:
                N = 0
            i += 1
            N += 1

        self.notify_on_complete()

        return self.solve_result
