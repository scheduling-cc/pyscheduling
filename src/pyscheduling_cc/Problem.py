from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass
class Instance(ABC):

    name: str

    @classmethod
    @abstractmethod
    def read_txt(path: Path):
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
    def generate_random(protocol: str = None):
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


class SolveStatus(Enum):
    INFEASIBLE = 1
    FEASIBLE = 2
    OPTIMAL = 3


@dataclass
class SolveResult:

    all_solutions = list[Solution]
    best_solution = Solution  # Needs to be consistent with "all_solutions" list
    solve_status = SolveStatus
    runtime = float
    kpis = dict[str, object]  # Other metrics that are problem / solver specific

    @property
    def nb_solutions(self) -> int:
        """Returns the number of solutions as an instance attribute (property)

        Returns:
            int: number of solutions
        """
        return len(self.all_solutions)
