import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyscheduling.core.listeners import BaseListener


@dataclass
class CSVListener(BaseListener):

    file_path: Path = field(default=Path("solver_log.csv"))
    _file: TextIO = field(init=False, repr=False)
    
    def on_start(self, solve_result, start_time):
        """Called at the start of the algorithm

        Args:
            start_time (int): time to start recording results
        """
        super().on_start(solve_result, start_time)
        self.columns = ["Info", "Iteration", "Time (s)", "Objective", "Best"]
        
        try:
            self._file = open(self.file_path, "w+", newline='')
            self.csv_writer = csv.DictWriter(self._file, fieldnames=self.columns)
            self.csv_writer.writeheader()
        except IOError:
            print(f"ERROR: cannot open the logging file {self.file_path}")
    
    def on_complete(self, end_time):
        """Called at the end of the algorithm

        Args:
            end_time (int):time at the end of the algorithm
        """
        super().on_complete(end_time)
        if self._file is not None:
            try:
                self._file.close()
            except IOError:
                print(f"ERROR: cannot close the logging file {self.file_path}")

    def on_solution_found(self, new_solution, time_found):
        """Called each time a new solution is found

        Args:
            new_solution : the solution found
            time_found : the time a solution is found
        """
        self._nb_sol += 1
        # Check if the new solution is the best one
        found_new_best = self.check_best_sol(new_solution)
        
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
        
        if self._file is not None:
            self.csv_writer.writerow(new_row)

            if self._nb_sol % 10 == 0:
                self._file.flush()
