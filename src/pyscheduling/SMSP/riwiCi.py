from time import perf_counter

import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Constraints, Objective, Solver
from pyscheduling.SMSP.SingleMachine import single_instance
import pyscheduling.SMSP.SingleMachine as SingleMachine
import pyscheduling.SMSP.SM_Methods as Methods
from pyscheduling.SMSP.SM_Methods import ExactSolvers


@single_instance([Constraints.W, Constraints.R], Objective.wiCi)
class riwiCi_Instance(SingleMachine.SingleInstance):
        
    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.WSECi



class Heuristics():
    @staticmethod
    def WSECi(instance : riwiCi_Instance):
        """Weighted Shortest Expected Completion time, dynamic dispatching rule inspired from WSPT but adds release
        time to processing time

        Args:
            instance (riwiCi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        startTime = perf_counter()
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
        return RootProblem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])

    @staticmethod
    def WSAPT(instance : riwiCi_Instance):
        """Weighted Shortest Available Processing time, dynamic dispatching rule inspired from WSPT but considers
        available jobs only at a given time t

        Args:
            instance (riwiCi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        startTime = perf_counter()
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
        return RootProblem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])

    @staticmethod
    def list_heuristic(instance : riwiCi_Instance, rule : int = 1):
        """contains a list of static dispatching rules to be chosen from

        Args:
            instance (riwiCi_Instance): Instance to be solved
            rule (int, optional) : Index of the rule to use. Defaults to 1.

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        startTime = perf_counter()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiCi_cache = []
        if rule==1: # Increasing order of the release time
            sorting_func = lambda job_id : instance.R[job_id]
            reverse = False
        elif rule==2: # WSPT
            sorting_func = lambda job_id : float(instance.W[job_id])/float(instance.P[job_id])
            reverse = True
        elif rule ==3: #WSPT including release time in the processing time
            sorting_func = lambda job_id : float(instance.W[job_id])/float(instance.R[job_id]+instance.P[job_id])
            reverse = True

        remaining_jobs_list = list(range(instance.n))
        remaining_jobs_list.sort(reverse=reverse,key=sorting_func)
        
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
        return RootProblem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]


class Metaheuristics(Methods.Metaheuristics):

    @staticmethod
    def iterative_LS(instance : riwiCi_Instance, ** kwargs):
        """Applies LocalSearch on the current solution iteratively

        Args:
            instance (riwiCi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        time_limit_factor = kwargs.get("time_limit_factor", None)
        init_sol_method = kwargs.get("init_sol_method", Heuristics.WSECi)
        Nb_iter = kwargs.get("Nb_iter", 500000)
        Non_improv = kwargs.get("Non_improv", 50000)

        first_time = perf_counter()
        if time_limit_factor:
            time_limit = instance.m * instance.n * time_limit_factor

        # Generate init solutoin using the initial solution method
        solution_init = init_sol_method(instance).best_solution

        if not solution_init:
            return RootProblem.SolveResult()

        local_search = SingleMachine.SM_LocalSearch()

        all_solutions = []
        solution_best = solution_init.copy()  # Save the current best solution
        solution_i = solution_init.copy()
        all_solutions.append(solution_best)

        N = 0
        i = 0
        time_to_best = perf_counter() - first_time
        while i < Nb_iter and N < Non_improv:
            # check time limit if exists
            if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                break

            solution_i = local_search.improve(solution_i,solution_i.instance.get_objective())

            if solution_i.objective_value < solution_best.objective_value:
                all_solutions.append(solution_i)
                solution_best = solution_i.copy()
                time_to_best = (perf_counter() - first_time)
                N = 0
            i += 1
            N += 1

        # Construct the solve result
        solve_result = RootProblem.SolveResult(
            best_solution=solution_best,
            solutions=all_solutions,
            runtime=(perf_counter() - first_time),
            time_to_best=time_to_best,
        )

        return solve_result

    
    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]
