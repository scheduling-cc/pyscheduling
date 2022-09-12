from math import exp
import random
from time import perf_counter

import pyscheduling.Problem as RootProblem
import pyscheduling.PMSP.ParallelMachines as pm

class Metaheuristics_Cmax():
    @staticmethod
    def meta_raps(instance: pm.ParallelInstance, p: float = 0.5, r: float = 0.5, n_iterations: int = 100):
        """Returns the solution using the meta-raps algorithm

        Args:
            instance (ParallelInstance): The instance to be solved by the metaheuristic
            p (float): probability of taking the greedy best solution
            r (float): percentage of moves to consider to select the best move
            n_iterations (int): Number of execution of the metaheuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = RootProblem.SolveResult()
        solveResult.all_solutions = []
        best_solution = None
        for _ in range(n_iterations):
            #solution = instance.create_solution()
            solution = pm.ParallelSolution(instance)
            remaining_jobs_list = [i for i in range(instance.n)]
            while len(remaining_jobs_list) != 0:
                insertions_list = []
                for i in remaining_jobs_list:
                    for j in range(instance.m):
                        current_machine_schedule = solution.machines[j]
                        insertions_list.append(
                            (i, j, 0, current_machine_schedule.completion_time_insert(i, 0, instance)))
                        for k in range(1, len(current_machine_schedule.job_schedule)):
                            insertions_list.append(
                                (i, j, k, current_machine_schedule.completion_time_insert(i, k, instance)))

                insertions_list = sorted(
                    insertions_list, key=lambda insertion: insertion[3])
                proba = random.random()
                if proba < p:
                    rand_insertion = insertions_list[0]
                else:
                    rand_insertion = random.choice(
                        insertions_list[0:int(instance.n * r)])
                taken_job, taken_machine, taken_pos, ci = rand_insertion
                solution.machines[taken_machine].job_schedule.insert(
                    taken_pos, pm.Job(taken_job, 0, 0))
                solution.machines[taken_machine].compute_completion_time(
                    instance, taken_pos)
                if taken_pos == len(solution.machines[taken_machine].job_schedule)-1:
                    solution.machines[taken_machine].last_job = taken_job
                if ci > solution.objective_value:
                    solution.objective_value = ci
                remaining_jobs_list.remove(taken_job)

            solution.fix_cmax()
            solveResult.all_solutions.append(solution)
            if not best_solution or best_solution.objective_value > solution.objective_value:
                best_solution = solution

        solveResult.best_solution = best_solution
        solveResult.runtime = perf_counter() - startTime
        solveResult.solve_status = RootProblem.SolveStatus.FEASIBLE
        return solveResult

    @staticmethod
    def grasp(instance: pm.ParallelInstance, x : float = 0.5, n_iterations: int = 100):
        """Returns the solution using the grasp algorithm

        Args:
            instance (ParallelInstance): Instance to be solved by the metaheuristic
            x (float): percentage of moves to consider to select the best move
            n_iterations (int): Number of execution of the metaheuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = RootProblem.SolveResult()
        solveResult.all_solutions = []
        best_solution = None
        for _ in range(n_iterations):
            #solution = instance.create_solution()
            solution = pm.ParallelSolution(instance)
            remaining_jobs_list = [i for i in range(instance.n)]
            while len(remaining_jobs_list) != 0:
                insertions_list = []
                for i in remaining_jobs_list:
                    for j in range(instance.m):
                        current_machine_schedule = solution.machines[j]
                        insertions_list.append(
                            (i, j, 0, current_machine_schedule.completion_time_insert(i, 0, instance)))
                        for k in range(1, len(current_machine_schedule.job_schedule)):
                            insertions_list.append(
                                (i, j, k, current_machine_schedule.completion_time_insert(i, k, instance)))

                insertions_list = sorted(
                    insertions_list, key=lambda insertion: insertion[3])
                rand_insertion = random.choice(
                    insertions_list[0:int(instance.n * x)])
                taken_job, taken_machine, taken_pos, ci = rand_insertion
                solution.machines[taken_machine].job_schedule.insert(
                    taken_pos, pm.Job(taken_job, 0, 0))
                solution.machines[taken_machine].compute_completion_time(
                    instance, taken_pos)
                if taken_pos == len(solution.machines[taken_machine].job_schedule)-1:
                    solution.machines[taken_machine].last_job = taken_job
                remaining_jobs_list.remove(taken_job)

            solution.fix_cmax()
            solveResult.all_solutions.append(solution)
            if not best_solution or best_solution.objective_value > solution.objective_value:
                best_solution = solution

        solveResult.best_solution = best_solution
        solveResult.runtime = perf_counter() - startTime
        solveResult.solve_status = RootProblem.SolveStatus.FEASIBLE
        return solveResult

    @staticmethod
    def lahc(instance: pm.ParallelInstance, **kwargs):
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
            return RootProblem.SolveResult()

        local_search = pm.PM_LocalSearch()

        if LS:
            solution_init = local_search.improve(
                solution_init)  # Improve it with LS

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

            solution_i = pm.NeighbourhoodGeneration.lahc_neighbour(
                current_solution)

            if LS:
                solution_i = local_search.improve(solution_i)
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
        solve_result = RootProblem.SolveResult(
            best_solution=solution_best,
            solutions=all_solutions,
            runtime=(perf_counter() - first_time),
            time_to_best=time_to_best,
        )

        return solve_result

    @staticmethod
    def SA(instance: pm.ParallelInstance, **kwargs):
        """ Returns the solution using the simulated annealing algorithm or the restricted simulated annealing algorithm
        
        Args:
            instance (ParallelInstance): Instance object to solve
            T0 (float, optional): Initial temperature. Defaults to 1.1.
            Tf (float, optional): Final temperature. Defaults to 0.01.
            k (float, optional): Acceptance facture. Defaults to 0.1.
            b (float, optional): Cooling factor. Defaults to 0.97.
            q0 (int, optional): Probability to apply restricted swap compared to restricted insertion. Defaults to 0.5.
            n_iterations (int, optional): Number of iterations for each temperature. Defaults to 10.
            Non_improv (int, optional): SA stops when the number of iterations without improvement is achieved. Defaults to 500.
            LS (bool, optional): Flag to apply local search at each iteration or not. Defaults to True.
            time_limit_factor: Fixes a time limit as follows: n*m*time_limit_factor if specified, else n_iterations is taken Defaults to None
            init_sol_method: The method used to get the initial solution. Defaults to "constructive"
            seed (int, optional): Seed for the random operators to make the algo deterministic if fixed. Defaults to None.

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """

        # Extracting the parameters
        restriced = kwargs.get("restricted", False)
        time_limit_factor = kwargs.get("time_limit_factor", None)
        init_sol_method = kwargs.get("init_sol_method", instance.init_sol_method())
        T0 = kwargs.get("T0", 1.4)
        Tf = kwargs.get("Tf", 0.01)
        k = kwargs.get("k", 0.1)
        b = kwargs.get("b", 0.99)
        q0 = kwargs.get("q0", 0.5)
        n_iterations = kwargs.get("n_iterations", 20)
        Non_improv = kwargs.get("Non_improv", 5000)
        LS = kwargs.get("LS", True)
        seed = kwargs.get("seed", None)

        if restriced:
            generationMethod = pm.NeighbourhoodGeneration.RSA_neighbour
            data = {'q0': q0}
        else:
            generationMethod = pm.NeighbourhoodGeneration.SA_neighbour
            data = {}
        if seed:
            random.seed(seed)

        first_time = perf_counter()
        if time_limit_factor:
            time_limit = instance.m * instance.n * time_limit_factor

        solution_init = init_sol_method(instance).best_solution

        if not solution_init:
            return RootProblem.SolveResult()

        local_search = pm.PM_LocalSearch()

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
            # check time limit if exists
            if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                break
            for i in range(0, n_iterations):
                # check time limit if exists
                if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                    break

                # solution_i = ParallelMachines.NeighbourhoodGeneration.generate_NX(solution_best)  # Generate solution in Neighbour
                solution_i = generationMethod(solution_best, **data)
                if LS:
                    # Improve generated solution using LS
                    solution_i = local_search.improve(solution_i)

                delta_cmax = solution_init.objective_value - solution_i.objective_value
                if delta_cmax >= 0:
                    solution_init = solution_i
                else:
                    r = random.random()
                    factor = delta_cmax / (k * T)
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

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

