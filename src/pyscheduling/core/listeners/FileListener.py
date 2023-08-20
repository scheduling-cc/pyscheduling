from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyscheduling.core.listeners import BaseListener


@dataclass
class FileListener(BaseListener):

    file_path: Path = field(default=Path("solver_log.txt"))
    _file: TextIO = field(init=False, repr=False)

    def on_start(self, solve_result, start_time):
        """Called at the start of the algorithm

        Args:
            start_time (int): time to start recording results
        """
        super().on_start(solve_result, start_time)
        try:
            self._file = open(self.file_path, "w+")
            self._file.write(
                f"{'Info':>15}  |"+
                f"{'Iteration':>10}  |"+
                f"{'Time (s)':>10}  |"+
                f"{'Objective':>10}  |"+
                f"{'Best':>10}" + "\n\n"
            )
        except IOError as e:
            print(f"ERROR: cannot open the logging file {self.file_path}")
            print(e)
    
    def on_complete(self, end_time):
        """Called at the end of the algorithm

        Args:
            end_time (int):time at the end of the algorithm
        """
        super().on_complete(end_time)
        if self._file is not None:
            self._file.write("\n" + '-' * (80) + "\n\n")
        
            self._file.write(f"{'Search completed':<20} : {self.solve_result.nb_solutions} solution(s) found \n"+
                            f"{'Status code':<20} : {self.solve_result.solve_status}\n\n")

            self._file.write(f"{'Best objective':<20} : {self.solve_result.best_solution.objective_value}\n"+
                            f"{'Found at time':<20} : {self.solve_result.time_to_best:.2f}\n\n")
            
            self._file.write(f"{'Total time':<20} : {self.solve_result.runtime:.2f}\n"+
                            f"{'Search speed':<20} : {self._search_speed:.2f} sol. / s \n\n")
            
            self._file.write("\n" + '-' * (80) + "\n\n")

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
        info = "##" if found_new_best else ''
        time_str = f"{time_found:.2f}"
        
        # Log to file
        if self._file is not None:
            self._file.write(
                    f"{info:>15}  |"+
                    f"{self._nb_sol:>10}  |"+
                    f"{time_str:>10}  |"+
                    f"{new_solution.objective_value:>10}  |"+
                    f"{self.solve_result.best_solution.objective_value:>10}" + "\n"
                )

            if self._nb_sol % 10 == 0:
                self._file.flush()
