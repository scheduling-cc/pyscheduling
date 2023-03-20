import random
import sys
from functools import partial
from math import exp
from time import perf_counter
from typing import Callable

import pyscheduling.FS.FlowShop as FS
import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Job

try:
    import docplex
    from docplex.cp.expression import INTERVAL_MAX
    from docplex.cp.model import CpoModel
    from docplex.cp.solver.cpo_callback import CpoCallback
except ImportError:
    pass

DOCPLEX_IMPORTED = True if "docplex" in sys.modules else False


class Heuristics():
    
    @staticmethod
    def dispatch_heuristic(instance : FS.FlowShopInstance, rule : Callable, reverse: bool = False):
        """Orders the jobs according to the rule (lambda function) and returns the schedule accordignly

        Args:
            instance (SingleInstance): Instance to be solved
            rule (Callable): a lambda function that defines the sorting criteria taking the instance and job_id as the parameters
            reverse (bool, optional): flag to sort in decreasing order. Defaults to False.

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        startTime = perf_counter()
        solution = FS.FlowShopSolution(instance)
        
        remaining_jobs_list = list(range(instance.n))
        sort_rule = partial(rule, instance)

        remaining_jobs_list.sort(key=sort_rule, reverse=reverse)
        solution.job_schedule = [Job(job_id, -1, -1) for job_id in remaining_jobs_list]
        solution.compute_objective()
        return RootProblem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])

    def BIBA(instance: FS.FlowShopInstance):
        """the greedy constructive heuristic (Best Insertion Based approach) to find an initial solution of flowshop instances 

        Args:
            instance (FlowShopInstance): Instance to be solved by the heuristic


        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = FS.FlowShopSolution(instance=instance)

        remaining_jobs_list = [j for j in range(instance.n)]

        while len(remaining_jobs_list) != 0:
            min_obj = None
            for i in remaining_jobs_list:

                start_time, end_time = solution.simulate_insert_last(i)
                new_obj = solution.simulate_insert_objective(i, start_time, end_time)

                if not min_obj or (min_obj > new_obj):
                    min_obj = new_obj
                    taken_job = i

            solution.job_schedule.append(Job(taken_job, 0, 0))
            solution.compute_objective(startIndex=len(solution.job_schedule) - 1)
            remaining_jobs_list.remove(taken_job)
        
        return RootProblem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

    @staticmethod
    def grasp(instance: FS.FlowShopInstance, p: float = 0.5, r: int = 0.5, n_iterations: int = 5):
        """Returns the solution using the Greedy randomized adaptive search procedure algorithm

        Args:
            instance (SingleInstance): The instance to be solved by the heuristic
            p (float): probability of taking the greedy best solution
            r (int): percentage of moves to consider to select the best move
            nb_exec (int): Number of execution of the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        startTime = perf_counter()
        solveResult = RootProblem.SolveResult()
        best_solution = None
        for _ in range(n_iterations):
            solution = FS.FlowShopSolution(instance)
            remaining_jobs_list = [i for i in range(instance.n)]
            while len(remaining_jobs_list) != 0:
                insertions_list = []
                for i in remaining_jobs_list:
                    start_time, end_time = solution.simulate_insert_last(i)
                    new_obj = solution.simulate_insert_objective(i, start_time, end_time)
                    insertions_list.append((i, new_obj))

                insertions_list.sort(key=lambda insertion: insertion[1])
                proba = random.random()
                if proba < p:
                    rand_insertion = insertions_list[0]
                else:
                    rand_insertion = random.choice(
                        insertions_list[0:int(instance.n * r)])
                
                taken_job, new_obj = rand_insertion
                solution.job_schedule.append(Job(taken_job, 0, 0))
                solution.compute_objective(startIndex=len(solution.job_schedule) - 1)
                remaining_jobs_list.remove(taken_job)

            solveResult.all_solutions.append(solution)
            if not best_solution or best_solution.objective_value > solution.objective_value:
                best_solution = solution

        solveResult.best_solution = best_solution
        solveResult.runtime = perf_counter() - startTime
        solveResult.solve_status = RootProblem.SolveStatus.FEASIBLE
        return solveResult

    def MINIT(instance : FS.FlowShopInstance):
        """Gupta's MINIT heuristic which is based on iteratively scheduling a new job at the end
        so that it minimizes the idle time at the last machine

        Args:
            instance (FlowShop.FlowShopInstance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        start_time = perf_counter()
        solution = FS.FlowShopSolution(instance=instance)

        #step 1 : Find pairs of jobs (job_i,job_j) which minimizes the idle time
        min_idleTime = None
        idleTime_ij_list = []
        for job_i in range(instance.n):
            for job_j in range(instance.n):
                if job_i != job_j :
                    pair = (job_i,job_j)
                    solution.job_schedule = [Job(job_i, 0, 0),Job(job_j, 0, 0)] 
                    solution.compute_objective()
                    idleTime_ij = solution.idle_time()
                    idleTime_ij_list.append((pair,idleTime_ij))
                    if min_idleTime is None or idleTime_ij<min_idleTime : min_idleTime = idleTime_ij

        min_IT_list = [pair_idleTime_couple for pair_idleTime_couple in idleTime_ij_list if pair_idleTime_couple[1] == min_idleTime]
        #step 2 : Break the tie by choosing the pair based on performance at increasingly earlier machines (m-2,m-3,..)
        # For simplicity purposes, a random choice is performed
        min_IT = random.choice(min_IT_list)
        
        i, j = min_IT[0] # Taken pair
        job_schedule = [Job(i, 0, 0), Job(j, 0, 0)]
        solution.job_schedule = job_schedule
        solution.compute_objective()
        #step 3 :
        remaining_jobs_list = [job_id for job_id in list(range(instance.n)) if job_id not in {i, j}]

        while len(remaining_jobs_list) > 0 :
            min_IT_factor = None
            old_idleTime = solution.idle_time()
            for job_id in remaining_jobs_list:
                last_job_startTime, new_cmax = solution.simulate_insert_last(job_id)
                factor = old_idleTime + (last_job_startTime - solution.machines[instance.m-1].objective_value)
                if min_IT_factor is None or factor < min_IT_factor:
                    min_IT_factor = factor
                    taken_job = job_id

            solution.job_schedule.append(Job(taken_job, 0, 0))
            remaining_jobs_list.remove(taken_job)
            solution.compute_objective(startIndex=len(job_schedule)-1)

        return RootProblem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

class Metaheuristics():

    @staticmethod
    def lahc(instance: FS.FlowShopInstance, **kwargs):
        """ Returns the solution using the LAHC algorithm
        Args:
            instance (ParallelInstance): Instance object to solve
            Lfa (int, optional): Size of the candidates list. Defaults to 25.
            n_iterations (int, optional): Number of iterations of LAHC. Defaults to 300.
            Non_improv (int, optional): LAHC stops when the number of iterations without improvement is achieved. Defaults to 50.
            LS (bool, optional): Flag to apply local search at each iteration or not. Defaults to True.
            time_limit_factor: Fixes a time limit as follows: n*m*time_limit_factor if specified, else n_iterations is taken Defaults to None
            init_sol_method: The method used to get the initial solution. Defaults to "constructive"
            seed (int, optional): Seed for the random operators to make the algo deterministic
        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        # Extracting parameters
        time_limit_factor = kwargs.get("time_limit_factor", None)
        init_sol_method = kwargs.get("init_sol_method", instance.init_sol_method())
        Lfa = kwargs.get("Lfa", 30)
        n_iterations = kwargs.get("n_iterations", 5000)
        Non_improv = kwargs.get("Non_improv", 500)
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
            return RootProblem.SolveResult()
        
        local_search = FS.FS_LocalSearch()

        if LS:
            solution_init = local_search.improve(solution_init)  # Improve it with LS

        all_solutions = []
        solution_best = solution_init.copy()  # Save the current best solution
        all_solutions.append(solution_best)
        lahc_list = [solution_init.objective_value for i in range(Lfa)]  # Create LAHC list

        N = 0
        i = 0
        time_to_best = perf_counter() - first_time
        current_solution = solution_init.copy()
        while i < n_iterations and N < Non_improv:
            # check time limit if exists
            if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                break
            
            solution_i = FS.NeighbourhoodGeneration.random_neighbour(current_solution)
            if LS:
                solution_i = local_search.improve(solution_i)
            if solution_i.objective_value < current_solution.objective_value or solution_i.objective_value < lahc_list[i % Lfa]:

                current_solution = solution_i
                if solution_i.objective_value < solution_best.objective_value:
                    all_solutions.append(solution_i)
                    solution_best = solution_i.copy()
                    time_to_best = (perf_counter() - first_time)
                    N = 0
            lahc_list[i % Lfa] = solution_i.objective_value
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

    def SA(instance: FS.FlowShopInstance, **kwargs):
        """ Returns the solution using the simulated annealing algorithm
        
        Args:
            instance (ParallelInstance): Instance object to solve
            T0 (float, optional): Initial temperature. Defaults to 1.1.
            Tf (float, optional): Final temperature. Defaults to 0.01.
            k (float, optional): Acceptance facture. Defaults to 0.1.
            b (float, optional): Cooling factor. Defaults to 0.97.
            q0 (int, optional): Probability to apply restricted swap compared to restricted insertion. Defaults to 0.5.
            n_iterations (int, optional): Number of iterations for each temperature. Defaults to 20.
            Non_improv (int, optional): SA stops when the number of iterations without improvement is achieved. Defaults to 500.
            LS (bool, optional): Flag to apply local search at each iteration or not. Defaults to True.
            time_limit_factor: Fixes a time limit as follows: n*m*time_limit_factor if specified, else n_iterations is taken Defaults to None
            init_sol_method: The method used to get the initial solution. Defaults to BIBA
            seed (int, optional): Seed for the random operators to make the algo deterministic if fixed. Defaults to None.

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """

        # Extracting the parameters
        time_limit_factor = kwargs.get("time_limit_factor", None)
        init_sol_method = kwargs.get("init_sol_method", instance.init_sol_method())
        T0 = kwargs.get("T0", 1.4)
        Tf = kwargs.get("Tf", 0.01)
        k = kwargs.get("k", 0.1)
        b = kwargs.get("b", 0.99)
        n_iterations = kwargs.get("n_iterations", 10)
        Non_improv = kwargs.get("Non_improv", 50)
        LS = kwargs.get("LS", True)
        seed = kwargs.get("seed", None)

        if seed:
            random.seed(seed)

        first_time = perf_counter()
        if time_limit_factor:
            time_limit = instance.n * time_limit_factor

        solution_init = init_sol_method(instance).best_solution

        if not solution_init:
            return RootProblem.SolveResult()

        local_search = FS.FS_LocalSearch()

        if LS:
            solution_init = local_search.improve(solution_init)

        all_solutions = []
        # Initialisation
        T = T0
        N = 0
        time_to_best = 0
        solution_i = None
        all_solutions.append(solution_init)
        solution_best = solution_init
        while T > Tf and (N != Non_improv):

            for i in range(0, n_iterations):
                # check time limit if exists
                if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                    break

                solution_i = FS.NeighbourhoodGeneration.deconstruct_construct(solution_init)
                if LS:
                    # Improve generated solution using LS
                    solution_i = local_search.improve(solution_i)

                delta_objective = solution_init.objective_value - solution_i.objective_value
                if delta_objective >= 0:
                    solution_init = solution_i
                else:
                    r = random.random()
                    factor = delta_objective / (k * T)
                    exponent = exp(factor)
                    if (r < exponent):
                        solution_init = solution_i

                if solution_best.objective_value > solution_init.objective_value:
                    all_solutions.append(solution_init)
                    solution_best = solution_init
                    time_to_best = (perf_counter() - first_time)
                    N = 0

            T = T * b
            N += 1

        # Construct the solve result
        solve_result = RootProblem.SolveResult(
            best_solution=solution_best,
            runtime=(perf_counter() - first_time),
            time_to_best=time_to_best,
            solutions=all_solutions
        )

        return solve_result

if DOCPLEX_IMPORTED:
    class CSP():

        CPO_STATUS = {
            "Feasible": RootProblem.SolveStatus.FEASIBLE,
            "Optimal": RootProblem.SolveStatus.OPTIMAL
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
        def _csp_transform_solution(msol, E_i, instance: FS.FlowShopInstance):

            sol = FS.FlowShopSolution(instance)
            for k in range(instance.m):
                k_tasks = []
                for i in range(instance.n):
                    start = msol[E_i[i][k]][0]
                    end = msol[E_i[i][k]][1]
                    k_tasks.append(Job(i,start,end))

                    k_tasks = sorted(k_tasks, key= lambda x: x[1])
                    sol.machines[k].oper_schedule = [job[0] for job in k_tasks]
            
            sol.job_schedule = sol.machines[0].oper_schedule
            
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
                objective = instance.get_objective()
                log_path = kwargs.get("log_path", None)
                time_limit = kwargs.get("time_limit", 300)
                nb_threads = kwargs.get("threads", 1)
                stop_times = kwargs.get(
                    "stop_times", [time_limit // 4, time_limit // 2, (time_limit * 3) // 4, time_limit])

                E = range(instance.n)
                M = range(instance.m)

                model = CpoModel("FS_Model")

                # Preparing transition matrices
                trans_matrix = {}
                if hasattr(instance, 'S'):
                    for k in range(instance.m):
                        k_matrix = [ [0 for _ in range(instance.n + 1)] for _ in range(instance.n + 1) ]
                        for i in range(instance.n):
                            ele = instance.S[k][i][i]
                            k_matrix[i+1][0] = ele
                            k_matrix[0][i+1] = ele

                            for j in range(instance.n):
                                k_matrix[i+1][j+1] = instance.S[k][i][j]

                        trans_matrix[k] = model.transition_matrix(k_matrix)
                    
                    # Create a dummy job for the first task
                    first_task = model.interval_var(size=0, optional= False, start = 0, name=f'first_task')

                E_i = [[] for i in E]
                M_k = [[] for k in M]
                types_k = [ list(range(1, instance.n + 1)) for k in M ]
                for i in E:
                    for k in M:
                        start_period = (instance.R[i], INTERVAL_MAX) if hasattr(instance, 'R') else (0, INTERVAL_MAX)
                        job_i = model.interval_var( start = start_period,
                                                    size = instance.P[i][k], optional= False, name=f'E[{i},{k}]')
                        E_i[i].append(job_i)
                        M_k[k].append(job_i)

                # No overlap inside machines
                seq_array = []
                for k in M:
                    if hasattr(instance, 'S'):
                        seq_k = model.sequence_var([first_task] + M_k[k], [0] + types_k[k], name=f"Seq_{k}")
                        model.add( model.no_overlap(seq_k, trans_matrix[k]) )
                    else:
                        seq_k = model.sequence_var(M_k[k], types_k[k], name=f"Seq_{k}")
                        model.add( model.no_overlap(seq_k) )
                        
                    seq_array.append(seq_k)
                    
                # Same sequence constraint
                for k in range(1, instance.m):
                    model.add( model.same_sequence(seq_array[k - 1], seq_array[k]) )

                # Precedence constraint between machines for each job
                for i in E:
                    for k in range(1, instance.m):
                        model.add( model.end_before_start(E_i[i][k - 1], E_i[i][k]) )

                # Add objective
                if objective == RootProblem.Objective.Cmax:
                    model.add( model.minimize( model.max(model.end_of(job_i) for i in E for job_i in E_i[i]) ) )
                elif objective == RootProblem.Objective.wiCi:
                    model.add(model.minimize( sum( instance.W[i] * model.end_of(E_i[i][-1]) for i in E ) )) # sum_{i in E} wi * ci
                elif objective == RootProblem.Objective.wiFi:
                    model.add(model.minimize( sum( instance.W[i] * (model.end_of(E_i[i][-1]) - instance.R[i]) for i in E ) )) # sum_{i in E} wi * (ci - ri)
                elif objective == RootProblem.Objective.wiTi:
                    model.add( model.minimize( 
                        sum( instance.W[i] * model.max(model.end_of(E_i[i]) - instance.D[i], 0) for i in E ) # sum_{i in E} wi * Ti
                    ))
                
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

                sol = CSP._csp_transform_solution(msol, E_i, instance)

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

                solve_result = RootProblem.SolveResult(
                    best_solution=sol,
                    runtime=msol.get_infos()["TotalTime"],
                    time_to_best=mycallback.best_sol_time,
                    status=CSP.CPO_STATUS.get(
                        msol.get_solve_status(), RootProblem.SolveStatus.INFEASIBLE),
                    kpis=kpis
                )

                return solve_result

            else:
                print("Docplex import error: you can not use this solver")

class ExactSolvers():

    if DOCPLEX_IMPORTED:
        @staticmethod
        def csp(instance, **kwargs):
            return CSP.solve(instance, **kwargs)
    else:
        @staticmethod
        def csp(instance, **kwargs):
            print("Docplex import error: you can not use this solver")
            return None
