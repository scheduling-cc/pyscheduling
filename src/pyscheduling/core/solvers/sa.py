from math import exp
import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from pyscheduling.core.solvers import Solver
from pyscheduling.Problem import BaseInstance, LocalSearch, SolveResult

class SA(Solver):

    # Params
    time_limit_factor : float = field(default=None) 
    init_sol_method: Solver = field(repr=False, default=None)
    init_temp: float = field(default=1.4)
    final_temp: float = field(default=0.01)
    k: float = field(default=0.1)
    cooling_factor: float = field(default=0.99)
    n_iterations: int = field(default=20)
    non_improv: int = field(default=500)
    use_local_search: bool = field(default=True)
    random_seed: int = field(default=None)
    
    # Required Operators
    ls_procedure: LocalSearch
    generate_neighbour: Callable

    def solve(self, instance: BaseInstance):
        """ Returns the solution using the simulated annealing algorithm or the restricted simulated annealing algorithm
        
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

        solution_init = self.init_sol_method(instance).best_solution

        if not solution_init:
            return SolveResult()

        if self.use_local_search:
            solution_init = self.ls_procedure.improve(solution_init)

        self.notify_on_solution_found(solution_init)

        # Initialisation
        T = self.init_temp
        N = 0
        solution_i = None
        curr_solution = solution_init.copy()
        best_solution = curr_solution
        while T > self.final_temp and (N != self.non_improv):
            
            for i in range(0, self.n_iterations):
                # check time limit if exists
                if use_time_limit and (perf_counter() - first_time) >= time_limit:
                    break

                solution_i = self.generate_neighbour(curr_solution) # **data
                if self.use_local_search:
                    # Improve generated solution using LS
                    solution_i = self.ls_procedure.improve(solution_i)

                delta_obj = curr_solution.objective_value - solution_i.objective_value
                if delta_obj >= 0:
                    curr_solution = solution_i.copy()
                else:
                    r = random.random()
                    factor = delta_obj / (self.k * T)
                    exponent = exp(factor)
                    if (r < exponent):
                        curr_solution = solution_i.copy()

                self.notify_on_solution_found(solution_i)

                if best_solution.objective_value > solution_i.objective_value:
                    N = 0

            T = T * self.cooling_factor
            N += 1
            
        self.notify_on_complete()
        return self.solve_result