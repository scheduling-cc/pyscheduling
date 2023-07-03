import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from pyscheduling.core.base_solvers.base_solver import BaseSolver
from pyscheduling.Problem import BaseInstance, LocalSearch, SolveResult


@dataclass
class BaseLAHC(BaseSolver):

    # Required Operators
    ls_procedure: LocalSearch
    generate_neighbour: Callable

    # Params
    time_limit_factor : float = field(default=None) 
    init_sol_method: BaseSolver = field(repr=False, default=None)
    history_list_size: int = field(default=30)
    n_iterations: int = field(default=5000)
    non_improv: int = field(default=500)
    use_local_search: bool = field(default=True)
    random_seed: int = field(default=None)
    
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
        if self.init_sol_method is None:
            solution_init = instance.init_sol_method.solve(instance).best_solution
        else:    
            solution_init = self.init_sol_method.solve(instance).best_solution
        
        if solution_init is None:
            return SolveResult()

        if self.use_local_search: 
            solution_init = self.ls_procedure.improve(solution_init)  # Improve it with LS
        
        self.notify_on_solution_found(solution_init)

        N = 0
        i = 0
        lahc_list = [solution_init.objective_value for _ in range(self.history_list_size)]  # Create LAHC list
        current_solution = solution_init.copy()  # Save the current best solution
        solution_best = current_solution
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
                    solution_best = solution_i
                    N = 0
            
            lahc_list[i % self.use_local_search] = solution_i.objective_value
            i += 1
            N += 1
        
        self.notify_on_complete()
        return self.solve_result
