import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from pyscheduling.core.solvers import Solver
from pyscheduling.Problem import BaseInstance, LocalSearch, SolveResult


@dataclass
class LAHC(Solver):

    # Params
    time_limit_factor : float = field(default=None) 
    init_sol_method: Solver = field(repr=False, default=None)
    history_list_size: int = field(default=30)
    n_iterations: int = field(default=5000)
    non_improv: int = field(default=500)
    use_local_search: bool = field(default=True)
    random_seed: int = field(default=None)
    
    # Required Operators
    ls_procedure: LocalSearch
    generate_neighbour: Callable

    def solve(self, instance: BaseInstance):
        """ Returns the solution using the LAHC algorithm

        Args:
            instance (ParallelInstance): Instance object to solve

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        random.seed(self.random_seed)

        first_time = perf_counter()
        use_time_limit = self.time_limit_factor is not None
        if use_time_limit:
            time_limit = instance.m * instance.n * self.time_limit_factor

        self.notify_on_start()

        # Generate init solutoin using the initial solution method
        solution_init = self.init_sol_method(instance).best_solution
        
        if solution_init is None:
            return SolveResult()

        if self.use_local_search: 
            solution_init = self.ls_procedure.improve(solution_init)  # Improve it with LS
            
        all_solutions = []
        solution_best = solution_init.copy()  # Save the current best solution
        all_solutions.append(solution_best)
        lahc_list = [solution_init.objective_value for _ in range(self.history_list_size)]  # Create LAHC list

        N = 0
        i = 0
        current_solution = solution_init.copy()
        while i < self.n_iterations and N < self.non_improv:
            # check time limit if exists
            if use_time_limit and (perf_counter() - first_time) >= time_limit:
                break

            solution_i = self.generate_neighbour(current_solution)
            
            if self.use_local_search:
                solution_i = self.ls_procedure.improve(solution_i)
            
            self.notify_on_solution_found(solution_i)

            if solution_i.objective_value < current_solution.objective_value or solution_i.objective_value < lahc_list[i % self.history_list_size]:

                current_solution = solution_i.copy()
                if solution_i.objective_value < solution_best.objective_value:
                    all_solutions.append(solution_i)
                    solution_best = solution_i.copy()
                    N = 0
            
            lahc_list[i % self.use_local_search] = solution_i.objective_value
            i += 1
            N += 1
        
        self.notify_on_complete()
        return self.solve_result
