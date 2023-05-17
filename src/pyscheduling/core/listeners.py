import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TextIO

from pyscheduling.Problem import SolveResult, SolveStatus


@dataclass
class BaseListener(ABC):

    solve_result : SolveResult = field(repr=False, init=False) 
    _start_time: int = field(init=False, repr=False)
    _end_time: int = field(init=False, repr=False)
    _total_time: int = field(init=False, repr=False)
    _nb_sol: int = 0
    
    def check_best_sol(self, solution, time_found):
        
        if self.solve_result.best_solution is None or solution.objective_value <= self.solve_result.best_solution.objective_value:
            self.solve_result.best_solution = solution

            if self.solve_result.time_to_best < time_found:
                self.solve_result.time_to_best = time_found

            return True

        return False

    def on_start(self, solve_result):
        self.solve_result = solve_result
        self._start_time = perf_counter()
    
    def on_complete(self):
        self._end_time = perf_counter()
        self._total_time = self._end_time - self._start_time
        self.solve_result.runtime = self._total_time
        self._search_speed = self._nb_sol / self.solve_result.runtime
        if self._nb_sol > 0:
            self.solve_result.solve_status = SolveStatus.FEASIBLE

    @abstractmethod
    def on_solution_found(self, new_solution, time_found, found_new_best):
        pass

@dataclass
class FileListener(BaseListener):

    file_path: Path = field(default=Path("solver_log.txt"))
    _file: TextIO = field(init=False, repr=False)

    def on_start(self, solve_result):
        super().on_start(solve_result)

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
        super().on_complete()

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
        # Check if the new solution is the best one
        found_new_best = self.check_best_sol(new_solution, time_found)
        
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


@dataclass
class CSVListener(BaseListener):

    file_path: Path = field(default=Path("solver_log.csv"))
    _file: TextIO = field(init=False, repr=False)
    
    def on_start(self, solve_result):
        super().on_start(solve_result)
        self.columns = ["Info", "Iteration", "Time (s)", "Objective", "Best"]
        
        try:
            self._file = open(self.file_path, "w+", newline='')
            self.csv_writer = csv.DictWriter(self._file, fieldnames=self.columns)
            self.csv_writer.writeheader()
        except IOError:
            print(f"ERROR: cannot open the logging file {self.file_path}")
    
    def on_complete(self):
        super().on_complete()
        if not self._file is None:
            try:
                self._file.close()
            except IOError:
                print(f"ERROR: cannot close the logging file {self.file_path}")

    def on_solution_found(self, new_solution):
        self._nb_sol += 1
        time_found = perf_counter() - self._start_time
        # Check if the new solution is the best one
        found_new_best = self.check_best_sol(new_solution, time_found)
        
        # Log parts
        info = "Found new best" if found_new_best else 'Found solution'
        time_str = f"{time_found:.2f}"

        new_row = {
            "Info": info,
            "Iteration": self._nb_sol,
            "Time (s)": time_str,
            "Objective": new_solution.objective_value,
            "Best": self.solve_result.best_solution.objective_value
        }

        self.csv_writer.writerow(new_row)

        if self._nb_sol % 10 == 0:
            self._file.flush()
