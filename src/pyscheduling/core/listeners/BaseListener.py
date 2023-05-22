from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pyscheduling.Problem import SolveResult, SolveStatus


@dataclass
class BaseListener(ABC):

    solve_result : SolveResult = field(repr=False, init=False) 
    _start_time: int = field(init=False, repr=False)
    _end_time: int = field(init=False, repr=False)
    _total_time: int = field(init=False, default=0)
    _nb_sol: int = field(init=False, default=0)
    
    def check_best_sol(self, solution):
        """Check if the new solution is the best found so far

        Args:
            solution (BaseSolution): Found solution

        Returns:
            bool: True if it is the best so far. False otherwise.
        """
        return self.solve_result.best_solution is None or solution.objective_value <= self.solve_result.best_solution.objective_value
    
    def on_start(self, solve_result, start_time):
        """Start Listening to the solve provess

        Args:
            solve_result (SolveResult): Solve result containing the solutions and other metrics
            start_time (int): timestamp of the start of the solve process
        """
        self.solve_result = solve_result
        self._start_time = start_time
    
    def on_complete(self, end_time):
        """Finish listerning to the solve process 

        Args:
            end_time (int): timestamp of the end of the solve process
        """
        self._end_time = end_time
        self._total_time = self._end_time - self._start_time
        self._search_speed = self._nb_sol / self.solve_result.runtime
        if self._nb_sol > 0:
            self.solve_result.solve_status = SolveStatus.FEASIBLE

    @abstractmethod
    def on_solution_found(self, new_solution, time_found):
        """Callback to finding a solution

        Args:
            new_solution (BaseSolution): Found solution
            time_found (int): timestamp of the moment the solution was found
        """
        pass