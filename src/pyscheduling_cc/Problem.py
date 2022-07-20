from abc import ABC, abstractmethod
from dataclasses import dataclass,field
from enum import Enum
from pathlib import Path


@dataclass
class Instance(ABC):

    name: str

    @classmethod
    @abstractmethod
    def read_txt(cls,path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            Instance:

        """
        pass

    @classmethod
    @abstractmethod
    def generate_random(cls,protocol: str = None):
        """Generate a random instance according to a predefined protocol

        Args:
            protocol (string): represents the protocol used to generate the instance

        Returns:
            Instance:
        """
        pass

    @abstractmethod
    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        pass


@dataclass
class Solution(ABC):

    instance: Instance
    objective_value : int

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

    @property
    def get_objective(self) -> int:
        """Return the objective value of the solution

        Returns:
            int: objective value
        """
        return self.objective_value

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


class SolveStatus(Enum):
    INFEASIBLE = 1
    FEASIBLE = 2
    OPTIMAL = 3


@dataclass
class SolveResult:

    all_solutions: list[Solution]
    best_solution: Solution  # Needs to be consistent with "all_solutions" list
    solve_status: SolveStatus
    runtime: float
    kpis: dict[str, object]  # Other metrics that are problem / solver specific

    def __init__(self,best_solution : Solution = None, runtime : float = -1,
                time_to_best : float = -1, status : SolveStatus = SolveStatus.FEASIBLE,
                solutions : list[Solution] = None,other_metrics : list[str,object] = None):
        
        self.best_solution = best_solution
        self.runtime = runtime
        if best_solution:
            self.solve_status = status
        else:
            self.status = "Infeasible"
        self.time_to_best = time_to_best
        self.other_metrics = other_metrics
        self.all_solutions = solutions

    @property
    def nb_solutions(self) -> int:
        """Returns the number of solutions as an instance attribute (property)

        Returns:
            int: number of solutions
        """
        return len(self.all_solutions)

    def __str__(self):
        return f'Search stopped with status : {self.solve_status.name}\n ' + \
                f'Solution is : \n {self.best_solution}s \n' + \
                f'Runtime is : {self.runtime}s \n'+ \
                f'time to best is : {self.time_to_best}s \n'

@dataclass
class LocalSearch():

    methods : list[object] = field(default_factory=list)
    copy_solution: bool = False  # by default for performance reasons

    @classmethod
    def instantiate(cls,methods : list[object] = None,copy_solution : bool = False):
        if methods is None: methods = cls.all_methods()
        for method in methods:
            if not callable(method):
                raise ValueError("Is not a function")
        return cls(methods,copy_solution)

    @classmethod
    def all_methods(cls):
        return [getattr(cls,func) for func in dir(cls) if not func.startswith("__") and func.startswith("_")]
    
    def improve(self, solution: Solution) -> Solution:
        """Improves a solution by iteratively calling local search operators

        Args:
            solution (Solution): current solution

        Returns:
            Solution: improved solution
        """
        curr_sol = solution.copy() if self.copy_solution else solution
        for method in self.methods:
            curr_sol = method(curr_sol)

        return curr_sol
    


@dataclass
class Solver(ABC):

    method: object

    def __init__(self,method : object) -> None:
        if not callable(method):
            raise ValueError("Is not a function")
        else: self.method=method

    def solve(self, instance: Instance, **data) -> SolveResult:
        """Solves the instance and returns the corresponding solve result

        Args:
            instance (Instance): instance to be solved

        Returns:
            SolveResult: object containing information about the solving process
                        and result
        """
        try:
           return self.method(instance,**data)
        except:
            print("Do correctly use the method as explain below :\n"+self.method.__doc__)
        pass
