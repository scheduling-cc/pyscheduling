from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
import Problem

@dataclass
class ParallelInstance(Problem.Instance,ABC):
    
    @classmethod
    @abstractmethod
    def read_txt(cls,path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            ParallelInstance:

        """
        pass

    @classmethod
    @abstractmethod
    def generate_random(cls,protocol: str = None):
        """Generate a random instance according to a predefined protocol

        Args:
            protocol (string): represents the protocol used to generate the instance

        Returns:
            ParallelInstance:
        """
        pass

    @abstractmethod
    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        pass

class Machine:

    def __init__(self,machine_num,completion_time=0,last_job=-1,job_schedule=[]):
        self.machine_num = machine_num
        self.completion_time = completion_time
        self.last_job = last_job
        self.job_schedule = job_schedule
    
    def __str__(self):
        return "M" + str(self.machine_num + 1) + " [" + ", ".join(map(str,self.job_schedule)) + " Ci : " + str(self.completion_time) + " ]"

    def __eq__(self,other):
        same_machine = other.machine_num == self.machine_num
        same_schedule = other.job_schedule == self.job_schedule
        return (same_machine and same_schedule)

    def copy(self):
        return Machine(self.machine_num,self.completion_time,self.last_job,list(self.job_schedule))
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    @staticmethod
    def fromDict(machine_dict):
        return Machine(machine_dict["machine_num"],machine_dict["completion_time"],machine_dict["last_job"],machine_dict["job_schedule"])


@dataclass
class ParallelSolution(Problem.Solution,ABC):

    Configuration : list(Machine)

    @classmethod
    @abstractmethod
    def read_txt(cls,path: Path):
        """Read a solution from a txt file

        Args:
            path (Path): path to the solution's txt file of type Path from pathlib

        Returns:
            Solution:
        """
        pass

    @abstractmethod
    def get_objective(self) -> int:
        """Return the objective value of the solution

        Returns:
            int: objective value
        """
        pass

    @abstractmethod
    def to_txt(self, path: Path) -> None:
        """Export the solution to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        pass

    @abstractmethod
    def plot(self) -> None:
        """Plot the solution in an appropriate diagram"""
        pass

    @abstractmethod
    def copy(self):
        """Return a copy to the current solution

        Returns:
            Solution: copy of the current solution
        """
        pass




@dataclass
class LSProcedure:

    operators: list[Problem.LSOperator]
    copy_solution: bool = False  # by default for performance reasons

    def improve(self, solution: Problem.Solution) -> Problem.Solution:
        """Improves a solution by iteratively calling local search operators

        Args:
            solution (Solution): current solution

        Returns:
            Solution: improved solution
        """
        curr_sol = solution.copy() if self.copy_solution else solution
        for operator in self.operators:
            curr_sol = operator.search(curr_sol)

        return curr_sol


@dataclass
class PaarallelGA(Problem.Solver,ABC):

    ls_procedure: LSProcedure

    @abstractmethod
    def solve(self, instance: Problem.Instance) -> Problem.SolveResult:
        """Solves the instance and returns the corresponding solve result

        Args:
            instance (Instance): instance to be solved

        Returns:
            SolveResult: object containing information about the solving process
                        and result
        """
        pass

@dataclass
class PaarallelSA(Problem.Solver,ABC):

    ls_procedure: LSProcedure

    @abstractmethod
    def solve(self, instance: Problem.Instance) -> Problem.SolveResult:
        """Solves the instance and returns the corresponding solve result

        Args:
            instance (Instance): instance to be solved

        Returns:
            SolveResult: object containing information about the solving process
                        and result
        """
        pass
