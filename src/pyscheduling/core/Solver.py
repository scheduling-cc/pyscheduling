from abc import ABC
from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Union

from pyscheduling.Problem import SolveResult
from pyscheduling.core.listeners import BaseListener


@dataclass
class Solver(ABC):

    listeners: List[BaseListener] = field(default_factory=list, init=False, repr=False)
    solve_result: SolveResult = field(default_factory=SolveResult, init=False)

    def attach_listeners(self, listener: Union[BaseListener, List[BaseListener]]):
        if isinstance(listener, list):
            self.listeners.extend(listener)
        #elif not isinstance(listener, BaseListener):
        #    raise TypeError("ERROR: listeners should be of type BaseListener or List[BaseListener]")
        else:
            self.listeners.append(listener)

    def add_solution(self, new_solution):
        self.solve_result.all_solutions.append(new_solution)

    def on_start(self):
        for listener in self.listeners:
            listener.on_start(self.solve_result)

    def on_complete(self):
        for listener in self.listeners:
            listener.on_complete()
    
    def on_solution_found(self, new_solution):
        self.add_solution(new_solution)
        for listener in self.listeners:
            listener.on_solution_found(new_solution)

