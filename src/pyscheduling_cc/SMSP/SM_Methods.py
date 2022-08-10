import random
import sys
from time import perf_counter

import pyscheduling_cc.Problem as Problem
import pyscheduling_cc.SMSP.SingleMachine as SingleMachine
from pyscheduling_cc.Problem import Solver
from pyscheduling_cc.SMSP.SingleMachine import Job, Objective

try:
    from docplex.cp.expression import INTERVAL_MAX
    from docplex.cp.model import CpoModel
    from docplex.cp.solver.cpo_callback import CpoCallback
except ImportError:
    pass

class Metaheuristics():

    @staticmethod
    def lahc(instance : SingleMachine.SingleInstance, **kwargs):
        """ Returns the solution using the LAHC algorithm
        Args:
            instance (SingleMachine.SingleInstance): Instance object to solve
            Lfa (int, optional): Size of the candidates list. Defaults to 25.
            n_iterations (int, optional): Number of iterations of LAHC. Defaults to 300.
            Non_improv (int, optional): LAHC stops when the number of iterations without
                improvement is achieved. Defaults to 50.
            LS (bool, optional): Flag to apply local search at each iteration or not.
                Defaults to True.
            time_limit_factor: Fixes a time limit as follows: n*m*time_limit_factor if specified, 
                else n_iterations is taken Defaults to None
            init_sol_method: The method used to get the initial solution. 
                Defaults to "WSECi"
            seed (int, optional): Seed for the random operators to make the algo deterministic
        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """

        # Extracting parameters
        time_limit_factor = kwargs.get("time_limit_factor", None)
        init_sol_method = kwargs.get("init_sol_method", instance.init_sol_method())
        Lfa = kwargs.get("Lfa", 30)
        n_iterations = kwargs.get("n_iterations", 500000)
        Non_improv = kwargs.get("Non_improv", 50000)
        LS = kwargs.get("LS", True)
        seed = kwargs.get("seed", None)

        if seed:
            random.seed(seed)

        first_time = perf_counter()
        if time_limit_factor:
            time_limit = instance.m * instance.n * time_limit_factor

        # Generate init solutoin using the initial solution method
        solution_init = init_sol_method(instance).best_solution

        if not solution_init:
            return Problem.SolveResult()

        local_search = SingleMachine.SM_LocalSearch()

        if LS:
            solution_init = local_search.improve(
                solution_init,instance.get_objective())  # Improve it with LS

        all_solutions = []
        solution_best = solution_init.copy()  # Save the current best solution
        all_solutions.append(solution_best)
        lahc_list = [solution_init.objective_value] * Lfa  # Create LAHC list

        N = 0
        i = 0
        time_to_best = perf_counter() - first_time
        current_solution = solution_init
        while i < n_iterations and N < Non_improv:
            # check time limit if exists
            if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                break

            solution_i = SingleMachine.NeighbourhoodGeneration.lahc_neighbour(
                current_solution,instance.get_objective())

            if LS:
                solution_i = local_search.improve(solution_i,instance.get_objective())
            if solution_i.objective_value < current_solution.objective_value or solution_i.objective_value < lahc_list[i % Lfa]:
                current_solution = solution_i
                if solution_i.objective_value < solution_best.objective_value:
                    all_solutions.append(solution_i)
                    solution_best = solution_i
                    time_to_best = (perf_counter() - first_time)
                    N = 0
            lahc_list[i % Lfa] = solution_i.objective_value
            i += 1
            N += 1

        # Construct the solve result
        solve_result = Problem.SolveResult(
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

class ExactSolvers():

    @staticmethod
    def csp(instance, **kwargs):
        return CSP.solve(instance, **kwargs)

class CSP():

    CPO_STATUS = {
        "Feasible": Problem.SolveStatus.FEASIBLE,
        "Optimal": Problem.SolveStatus.OPTIMAL
    }

    class MyCallback(CpoCallback):

        def __init__(self, stop_times=[300, 600, 3600, 7200]):
            self.stop_times = stop_times
            self.best_values = dict()
            self.stop_idx = 0
            self.best_sol_time = 0
            self.nb_sol = 0

        def invoke(self, solver, event, jsol):

            if event == "Solution":
                self.nb_sol += 1
                solve_time = jsol.get_info('SolveTime')
                self.best_sol_time = solve_time

                # Go to the next stop time
                while self.stop_idx < len(self.stop_times) and solve_time > self.stop_times[self.stop_idx]:
                    self.stop_idx += 1

                if self.stop_idx < len(self.stop_times):
                    # Get important elements
                    obj_val = jsol.get_objective_values()[0]
                    self.best_values[self.stop_times[self.stop_idx]] = obj_val

    @staticmethod
    def _csp_transform_solution(msol, E_i, instance, objective : Objective):

        sol = instance.create_solution()
        k_tasks = []
        for i in range(instance.n):
            start = msol[E_i[i]][0]
            end = msol[E_i[i]][1]
            k_tasks.append(Job(i,start,end))
            
            k_tasks = sorted(k_tasks, key= lambda x: x[1])
            sol.machine.job_schedule = k_tasks
        
        if objective == Objective.wiCi:
            sol.wiCi()
        elif objective == Objective.wiTi:
            sol.wiTi()
        elif objective == Objective.Cmax:
            sol.cmax()

        return sol
    
    @staticmethod
    def solve(instance, **kwargs):
        """ Returns the solution using the Cplex - CP optimizer solver

        Args:
            instance (Instance): Instance object to solve
            objective (str): The objective to optimize. Defaults to wiCi
            log_path (str, optional): Path to the log file to output cp optimizer log. Defaults to None to disable logging.
            time_limit (int, optional): Time limit for executing the solver. Defaults to 300s.
            threads (int, optional): Number of threads to set for cp optimizer solver. Defaults to 1.

        Returns:
            SolveResult: The object represeting the solving process result
        """
        if "docplex" in sys.modules:
            # Extracting parameters
            objective = kwargs.get("objective", "wiCi")
            log_path = kwargs.get("log_path", None)
            time_limit = kwargs.get("time_limit", 300)
            nb_threads = kwargs.get("threads", 1)
            stop_times = kwargs.get(
                "stop_times", [time_limit // 4, time_limit // 2, (time_limit * 3) // 4, time_limit])

            E = range(instance.n)

            # Construct the model
            model = CpoModel("smspModel")

            # Jobs interval_vars including the release date and processing times constraints
            E_i = []
            for i in E:
                start_period = (instance.R[i], INTERVAL_MAX) if hasattr(instance, 'R') else (0, INTERVAL_MAX)
                job_i = model.interval_var( start = start_period,
                                            size = instance.P[i], optional= False, name=f'E[{i}]')
                E_i.append(job_i)

            # Sequential execution on the machine
            machine_sequence = model.sequence_var( E_i, list(E) )
            model.add( model.no_overlap(machine_sequence) )
            
            # Define the objective 
            if objective == Objective.wiCi:
                model.add(model.minimize( sum( instance.W[i] * model.end_of(E_i[i]) for i in E ) )) # sum_{i in E} wi * ci
            elif objective == Objective.wiTi:
                model.add( model.minimize( 
                    sum( instance.W[i] * model.max(model.end_of(E_i[i]) - instance.D[i], 0) for i in E ) # sum_{i in E} wi * Ti
                ))
            elif objective == Objective.Cmax:
                model.add(model.minimize( max( model.end_of(E_i[i]) for i in E ) )) # max_{i in E} ci 

            # Link the callback to save stats of the solve process
            mycallback = CSP.MyCallback(stop_times=stop_times)
            model.add_solver_callback(mycallback)

            # Run the model
            msol = model.solve(LogVerbosity="Normal", Workers=nb_threads, TimeLimit=time_limit, LogPeriod=1000000,
                               log_output=True, trace_log=False, add_log_to_solution=True, RelativeOptimalityTolerance=0)

            # Logging solver's infos if log_path is specified
            if log_path:
                with open(log_path, "a") as logFile:
                    logFile.write('\n\t'.join(msol.get_solver_log().split("!")))
                    logFile.flush()

            sol = CSP._csp_transform_solution(msol, E_i, instance, objective)

            # Construct the solve result
            kpis = {
                "ObjValue": msol.get_objective_value(),
                "ObjBound": msol.get_objective_bounds()[0],
                "MemUsage": msol.get_infos()["MemoryUsage"]
            }
            prev = -1
            for stop_t in mycallback.stop_times:
                if stop_t in mycallback.best_values:
                    kpis[f'Obj-{stop_t}'] = mycallback.best_values[stop_t]
                    prev = mycallback.best_values[stop_t]
                else:
                    kpis[f'Obj-{stop_t}'] = prev

            solve_result = Problem.SolveResult(
                best_solution=sol,
                runtime=msol.get_infos()["TotalTime"],
                time_to_best=mycallback.best_sol_time,
                status=CSP.CPO_STATUS.get(
                    msol.get_solve_status(), Problem.SolveStatus.INFEASIBLE),
                kpis=kpis
            )

            return solve_result

        else:
            print("Docplex import error: you can not use this solver")

class Heuristics_HelperFunctions():

    @staticmethod
    def rule_candidate(remaining_jobs : list[int], rule : object, reverse : bool = True):
        """Extract the highest index job using the specific passed rule.

        Args:
            remaining_jobs (list[int]): The list of jobs on which we apply the rule
            rule (object): The rule (function) which is used in the heuristic in order to extract the candidate job
            reverse (bool, optional): When true, the candidate returned is the job with the highest value returned by the rule.
            When false, returns the job with the lowest value returned by the rule. Defaults to True.

        Returns:
            int: returns the job candidate by the given rule from remaining_jobs
        """
        max_rule_value = -1
        min_rule_value = None
        for job in remaining_jobs:
            rule_value = rule(job)
            if max_rule_value<rule_value: 
                max_rule_value = rule_value
                taken_job_max = job
            if min_rule_value is None or min_rule_value>rule_value:
                min_rule_value = rule_value
                taken_job_min = job
        if reverse: return taken_job_max
        else: return taken_job_min

class Heuristics_Cmax():
    @staticmethod
    def meta_raps(instance: SingleMachine.SingleInstance, p: float, r: int, nb_exec: int):
        """Returns the solution using the meta-raps algorithm

        Args:
            instance (SingleInstance): The instance to be solved by the heuristic
            p (float): probability of taking the greedy best solution
            r (int): percentage of moves to consider to select the best move
            nb_exec (int): Number of execution of the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        pass

    @staticmethod
    def grasp(instance: SingleMachine.SingleInstance, x, nb_exec: int):
        """Returns the solution using the grasp algorithm

        Args:
            instance (SingleInstance): Instance to be solved by the heuristic
            x (_type_): percentage of moves to consider to select the best move
            nb_exec (int): Number of execution of the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        pass

 