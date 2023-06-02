from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Union

from pyscheduling.Problem import BaseInstance, SolveResult, BaseSolution
from pyscheduling.core.listeners import BaseListener


@dataclass
class Solver(ABC):

    listeners: List[BaseListener] = field(default_factory=list, init=False, repr=False)
    solve_result: SolveResult = field(default_factory=SolveResult, init=False)

    def attach_listeners(self, *listeners: BaseListener):
        """Subscribe a list of listeners to the solving process

        Raises:
            TypeError: if one of the passed arguments is not a subclass of BaseListener
        """
        for listener in listeners:
            if isinstance(listener, BaseListener):
                self.listeners.append(listener)
            else:        
                raise TypeError("ERROR: listeners should be of type BaseListener or List[BaseListener]")

    def add_solution(self, new_solution: BaseSolution, time_found: int):
        """Adds the new found solution to the solve_result and compute the current timestamp

        Args:
            new_solution (Solution): Found solution
            time_found (int): Timestamp of the moment the solution was found
        """
        self.solve_result.all_solutions.append(new_solution)
        if self.solve_result.best_solution is None or \
            new_solution.objective_value <= self.solve_result.best_solution.objective_value:
            self.solve_result.best_solution = new_solution
            self.solve_result.time_to_best = time_found

    def notify_on_start(self):
        """Notify the subscribed listeners of the start of the solve process
        """
        self._start_time = perf_counter()
        for listener in self.listeners:
            listener.on_start(self.solve_result, self._start_time)

    def notify_on_complete(self):
        """Notify the subscribed listeners of the end of the solve process
        """
        self._end_time = perf_counter()
        self.solve_result.runtime = self._end_time - self._start_time
        for listener in self.listeners:
            listener.on_complete(self._end_time)
    
    def notify_on_solution_found(self, new_solution: BaseSolution):
        """Notify the subscribe listeners of the new found solution

        Args:
            new_solution (BaseSolution): Found solution
        """
        time_found = perf_counter() - self._start_time
        self.add_solution(new_solution, time_found)
        for listener in self.listeners:
            listener.on_solution_found(new_solution, time_found)

    @abstractmethod
    def solve(self, instance: BaseInstance):
        pass

