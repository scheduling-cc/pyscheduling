from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class Objective(Enum):# Negative value are for minimization problems, Positive values are for maximization problems
    Cmax = -1
    wiTi = -2
    wiCi = -3
    Lmax = -4

    @classmethod
    def to_string(cls):
        """Print the available objective functions

        Returns:
            str: name of every objective in different lines
        """
        return cls.Cmax.name + "\n" + cls.wiTi.name + "\n" + cls.wiCi.name

@dataclass
class Instance(ABC):

    name: str

    @classmethod
    @abstractmethod
    def read_txt(cls, path: Path):
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
    def generate_random(cls, protocol: str = None):
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
    objective_value: int

    @classmethod
    @abstractmethod
    def read_txt(cls, path: Path):
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
    time_to_best: float
    solve_status: SolveStatus
    runtime: float
    kpis: dict[str, object]  # Other metrics that are problem / solver specific

    def __init__(self, best_solution: Solution = None, runtime: float = -1,
                 time_to_best: float = -1, status: SolveStatus = SolveStatus.FEASIBLE,
                 solutions: list[Solution] = None, kpis: list[str, object] = None):
        """constructor of SolveResult

        Args:
            best_solution (Solution, optional): Best solution among solutions. Defaults to None.
            runtime (float, optional): Execution time. Defaults to -1.
            time_to_best (float, optional): Estimated time left to find the best solution. Defaults to -1.
            status (SolveStatus, optional): Status of the solution. Defaults to SolveStatus.FEASIBLE.
            solutions (list[Solution], optional): All feasible solution of the problem. Defaults to None.
            other_metrics (list[str,object], optional): Supplementary information. Defaults to None.
        """
        self.best_solution = best_solution
        self.runtime = runtime
        if best_solution:
            self.solve_status = status
        else:
            self.solve_status = SolveStatus.INFEASIBLE
        self.time_to_best = time_to_best
        self.kpis = kpis
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
            f'Solution is : \n {self.best_solution} \n' + \
            f'Runtime is : {self.runtime}s \n' + \
            f'time to best is : {self.time_to_best}s \n'


@dataclass
class LocalSearch():

    methods: list[object] = field(default_factory=list)
    copy_solution: bool = False  # by default for performance reasons

    def __init__(self, methods: list[object] = None, copy_solution: bool = False):
        """Constructor of LocalSearch

        Args:
            methods (list[object], optional): List of methods or operators to be used in a given order. Defaults to None.
            copy_solution (bool, optional): if True, the original solution will not be altered but a new one will be created. Defaults to False.

        Raises:
            ValueError: if an element of methods is not a function
        """
        if methods is None:
            methods = self.all_methods()
        for method in methods:
            if not callable(method):
                raise ValueError("Is not a function")
        self.methods = methods
        self.copy_solution = copy_solution

    @classmethod
    def all_methods(cls):
        """returns all the methods of a given LocalSearch class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and func.startswith("_")]

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
class Branch_Bound():
    instance : Instance
    root : object = None
    objective_value = None
    best_solution : Solution = None
    all_solution : list[Solution] = field(default_factory=list)

    @dataclass
    class Node():
        lower_bound : float = None
        if_solution : bool = False
        partial_solution : object = None
        sub_nodes : list[object] = field(default_factory=list)

        def delete(self):
            """To delete the variable definitely
            """
            for node in self.sub_nodes : node.delete()
            del self

    def branch(self, node : Node):
        """branching strategy, to be redefined

        Args:
            node (Node): node to branch from
        """
        pass
    
    def bound(self, node : Node):
        """bounding method, to be redefined

        Args:
            node (Node): node to bound
        """
        pass
    
    def discard(self, root : Node, best_solution : float, objective : Objective):
        """prunes the search tree sections where we are certain a better solution will not be find there

        Args:
            root (Node): root node of the tree
            best_solution (float): best objective value
            objective (Objective): objective to be considered, to know if it's a minimization or a maximization problem
        """
        if root.lower_bound is not None:
            if objective.value > 0 and root.lower_bound < best_solution : root = None
            elif objective.value < 0 and root.lower_bound > best_solution : root = None
        for node in root.sub_nodes :
            if objective.value > 0 and node.lower_bound < best_solution : node = None
            elif objective.value < 0 and node.lower_bound > best_solution : node = None

    def objective(self, node : Node):
        """objective value evaluator, to be redefined

        Args:
            node (Node): node to be evaluated as a complete solution
        """
        pass

    def solve(self, root : Node = None):
        """recursive function to perform Branch&Bound on the instance attribute

        Args:
            root (Node, optional): starting node. Defaults to None.
        """
        if root is None : 
            root = self.Node()
            self.root = root
        self.branch(root) 
        if root.sub_nodes[0].if_solution is False :
            for node in root.sub_nodes: self.bound(node)
            sorted_sub_nodes = root.sub_nodes
            sorted_sub_nodes.sort(reverse= self.instance.get_objective().value > 0, key = lambda node : node.lower_bound)
            for node in sorted_sub_nodes : self.solve(node)
        else :
            for node in root.sub_nodes: 
                node.lower_bound = self.objective(node)
                self.all_solution.append(node.partial_solution)
                if self.best_solution is None or (self.instance.get_objective().value > 0 and self.objective_value < node.lower_bound) :
                    self.best_solution = node.partial_solution
                    self.objective_value = node.lower_bound
                elif self.best_solution is None or (self.instance.get_objective().value < 0 and node.lower_bound < self.objective_value) :
                    self.best_solution = node.partial_solution
                    self.objective_value = node.lower_bound
                self.discard(self.root,self.objective_value,self.instance.get_objective())
                
                    

@dataclass
class Solver(ABC):

    method: object

    def __init__(self, method: object) -> None:
        """_summary_

        Args:
            method (object): the function (heuristic/metaheuristic) that will be used to solve the problem

        Raises:
            ValueError: if an element of methods is not a function
        """
        if not callable(method):
            raise ValueError("Is not a function")
        else:
            self.method = method

    def solve(self, instance: Instance, **data) -> SolveResult:
        """Solves the instance and returns the corresponding solve result

        Args:
            instance (Instance): instance to be solved

        Returns:
            SolveResult: object containing information about the solving process
                        and result
        """
        try:
            return self.method(instance, **data)
        except:
            print("Do correctly use the method as explained below :\n" +
                  self.method.__doc__)
        pass
