from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TextIO

from pyscheduling.Problem import SolveResult

@dataclass
class BaseListener(ABC):

    solve_result : SolveResult = field(repr=False, init=False) 
    _start_time: int = field(init=False, repr=False)
    _end_time: int = field(init=False, repr=False)
    _total_time: int = field(init=False, repr=False)
    
    def save_solution(self, solution, time_found):
        self.solve_result.all_solutions.append(solution)
        if self.solve_result.best_solution is None or solution.objective_value < self.solve_result.best_solution.objective_value:
            self.solve_result.best_solution = solution

            if self.solve_result.time_to_best < time_found:
                self.solve_result.time_to_best = time_found

            return True

        return False

    @abstractmethod
    def on_start(self):
        pass
    
    @abstractmethod
    def on_complete(self):
        pass

    @abstractmethod
    def on_solution_found(self):
        pass

@dataclass
class FileListener(BaseListener):

    file_path: Path = field(default=Path("solver_log.txt"))
    _file: TextIO = field(init=False, repr=False)
    _nb_sol: int = 0 

    def on_start(self, solve_result):
        self.solve_result = solve_result
        self._start_time = perf_counter()
        try:
            self._file = open(self.file_path, "w+")
            self._file.write(
                f"{'Info':>15}  |"+
                f"{'Iteration':>10}  |"+
                f"{'Time (s)':>10}  |"+
                f"{'Objective':>10}  |"+
                f"{'Best':>10}" + "\n\n"
            )
        except IOError:
            print(f"ERROR: cannot open the logging file {self.file_path}")
    
    def on_complete(self):
        self._end_time = perf_counter()
        self._total_time = self._end_time - self._start_time
        self.solve_result.runtime = self._total_time
        self._search_speed = self._nb_sol / self.solve_result.runtime

        self._file.write("\n" + '-' * (55 + 5) + "\n\n")
        
        self._file.write(f"{'Search completed':<20} : {self.solve_result.nb_solutions} solution(s) found \n"+
                         f"{'Status code':<20} : {self.solve_result.solve_status}\n\n")

        self._file.write(f"{'Best objective':<20} : {self.solve_result.best_solution.objective_value}\n"+
                         f"{'Found at time':<20} : {self.solve_result.time_to_best:.2f}\n\n")
        
        self._file.write(f"{'Total time':<20} : {self.solve_result.runtime:.2f}\n"+
                         f"{'Search speed':<20} : {self._search_speed:.2f} sol. / s \n\n")
        
        self._file.write("\n" + '-' * (80) + "\n\n")

        if not self._file is None:
            try:
                self._file.close()
            except IOError:
                print(f"ERROR: cannot close the logging file {self.file_path}")

    def on_solution_found(self, new_solution):
        self._nb_sol += 1
        time_found = perf_counter() - self._start_time
        # Add the solution to solve result
        found_new_best = self.save_solution(new_solution, time_found)
        
        # Log parts
        info = "##" if found_new_best else ''
        time_str = f"{time_found:.2f}"

        # Log to file
        self._file.write(
                f"{info:>15}  |"+
                f"{self._nb_sol:>10}  |"+
                f"{time_str:>10}  |"+
                f"{new_solution.objective_value:>10}  |"+
                f"{self.solve_result.best_solution.objective_value:>10}" + "\n"
            )

        if self._nb_sol % 10 == 0:
            self._file.flush()


