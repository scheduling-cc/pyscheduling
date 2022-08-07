import random
from time import perf_counter

import pyscheduling_cc.Problem as Problem
from pyscheduling_cc.Problem import Solver
import pyscheduling_cc.SMSP.SingleMachine as SingleMachine

class Metaheuristics():

    @staticmethod
    def lahc(instance : SingleMachine.SingleInstance, **kwargs):
        """ Returns the solution using the LAHC algorithm
        Args:
            instance (SingleMachine.SingleInstance): Instance object to solve
            Lfa (int, optional): Size of the candidates list. Defaults to 25.
            Nb_iter (int, optional): Number of iterations of LAHC. Defaults to 300.
            Non_improv (int, optional): LAHC stops when the number of iterations without
                improvement is achieved. Defaults to 50.
            LS (bool, optional): Flag to apply local search at each iteration or not.
                Defaults to True.
            time_limit_factor: Fixes a time limit as follows: n*m*time_limit_factor if specified, 
                else Nb_iter is taken Defaults to None
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
        Nb_iter = kwargs.get("Nb_iter", 500000)
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
        while i < Nb_iter and N < Non_improv:
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

class Heuristics_HelperFunctions():

    @staticmethod
    def rule_candidate(remaining_jobs : list[SingleMachine.Job], rule : object, reverse : bool = True):
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

 