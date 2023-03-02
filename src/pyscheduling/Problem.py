from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Dict, List

Job = namedtuple('Job', ['id', 'start_time', 'end_time'])

class GenerationLaw(Enum):
    UNIFORM = 1
    NORMAL = 2

class Constraints(Enum):
    W = "weight"
    R = "release"
    S = "setup"
    D = "due"

    @classmethod
    def to_string(cls):
        """Print the available constraints for Single Machine

        Returns:
            str: name of every constraint in different lines
        """
        return cls.W.value + "\n" + cls.R.value + "\n" + cls.S.value + "\n" + cls.D.value

    def __lt__(self, other):
        """redefine less than operator alphabetically

        Args:
            other (Constraints): Another constraint

        Returns:
            bool: returns the comparison result
        """
        return self.name < other.name


class Objective(Enum):  # Negative value are for minimization problems, Positive values are for maximization problems
    Cmax = -1
    wiTi = -2
    wiCi = -3
    Lmax = -4
    wiFi = -5

    @classmethod
    def to_string(cls):
        """Print the available objective functions

        Returns:
            str: name of every objective in different lines
        """
        return "\n".join([e.name for e in cls]) 


class DecoratorsHelper():

    @staticmethod
    def set_new_attr(cls, name, value):
        """helper function to add a new function to the class if the user doesn't define it

        Args:
            name (str): name of the function
            value (Callable): function definition

        Returns:
            bool: True if the function is already defined from the user
        """
        if name in cls.__dict__:  # To allow overriding the default functions' implementation
            return True
        setattr(cls, name, value)
        return False

    @staticmethod
    def repr_fn(self):
        """__repr__ default function, returns a string of the class name and the fields of the instance

        Returns:
            str: string representation
        """
        return self.__class__.__qualname__ + \
            '\n'.join([f"({name}={value})"
                       for name, value in vars(self).items()])

    @staticmethod
    def str_fn(self):
        """__str__ default function, returns the object representation

        Returns:
            str: string representation
        """
        return self.__repr__()

    @staticmethod
    def update_abstractmethods(cls):
        """
        Ref: https://github.com/python/cpython/blob/6da1a2e993c955aa69158871b8c8792cef3094c3/Lib/abc.py#L146
        Recalculate the set of abstract methods of an abstract class.
        If a class has had one of its abstract methods implemented after the
        class was created, the method will not be considered implemented until
        this function is called. Alternatively, if a new abstract method has been
        added to the class, it will only be considered an abstract method of the
        class after this function is called.
        This function should be called before any use is made of the class,
        usually in class decorators that add methods to the subject class.
        Returns cls, to allow usage as a class decorator.
        If cls is not an instance of ABCMeta, does nothing.
        """
        if not hasattr(cls, '__abstractmethods__'):
            # We check for __abstractmethods__ here because cls might by a C
            # implementation or a python implementation (especially during
            # testing), and we want to handle both cases.
            return cls

        abstracts = set()
        # Check the existing abstract methods of the parents, keep only the ones
        # that are not implemented.
        for scls in cls.__bases__:
            for name in getattr(scls, '__abstractmethods__', ()):
                value = getattr(cls, name, None)
                if getattr(value, "__isabstractmethod__", False):
                    abstracts.add(name)
        # Also add any other newly added abstract methods.
        for name, value in cls.__dict__.items():
            if getattr(value, "__isabstractmethod__", False):
                abstracts.add(name)
        cls.__abstractmethods__ = frozenset(abstracts)
        return cls


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

    @staticmethod
    def read_1D(content: List[str], startIndex: int):
        """Read a table from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the vector

        Returns:
           (list[int],int): (vector, index of the next section of the instance)
        """
        i = startIndex + 1
        line = content[i].strip().split('\t')
        vector = []  # Table : Processing time of job i
        for j in line:
            vector.append(int(j))
        return (vector, i+1)

    @staticmethod
    def read_2D(dimension_i : int, content: List[str], startIndex: int):
        """Read a matrix from a list of lines extracted from the file of the instance

        Args:
            dimension_i (int): number of lines of the matrix, usually number of jobs 'n'.
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the matrix

        Returns:
           (list[list[int]],int): (Matrix, index of the next section of the instance)
        """
        i = startIndex
        Matrix = []  # Matrix S_ijk : Setup time between jobs j and k
        i += 1  # Skip SSD
        for k in range(dimension_i):
            line = content[i].strip().split('\t')
            Matrix_i = [int(val_str) for val_str in line]
            Matrix.append(Matrix_i)
            i += 1
        return (Matrix, startIndex+1+dimension_i)

    @staticmethod
    def read_3D(dimension_i : int, dimension_j : int, content: List[str], startIndex: int):
        """Read the table of matrices from a list of lines extracted from the file of the instance

        Args:
            dimension_i (int): Dimension of the table, usually number of machines 'm'.
            dimension_j (int): Dimension of the matrix, usually number of jobs 'n'.
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the table of matrices

        Returns:
           (list[list[list[int]]],int): (Table of matrices, index of the next section of the instance)
        """
        i = startIndex
        S = []  # Table of Matrix S_ijk : Setup time between jobs j and k on machine i
        i += 1  # Skip SSD
        endIndex = startIndex+1+dimension_j*dimension_i+dimension_i
        while i != endIndex:
            i = i+1  # Skip Mk
            Si = []
            for k in range(dimension_j):
                ligne = content[i].strip().split('\t')
                Sij = [int(val_str) for val_str in ligne]
                Si.append(Sij)
                i += 1
            S.append(Si)
        return (S, i)


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

    all_solutions: List[Solution]
    best_solution: Solution  # Needs to be consistent with "all_solutions" list
    time_to_best: float
    solve_status: SolveStatus
    runtime: float
    kpis: Dict[str, object]  # Other metrics that are problem / solver specific

    def __init__(self, best_solution: Solution = None, runtime: float = -1,
                 time_to_best: float = -1, status: SolveStatus = SolveStatus.FEASIBLE,
                 solutions: List[Solution] = None, kpis: Dict[str, object] = None):
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
        self.all_solutions = solutions if solutions is not None else []

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

    methods: List[object] = field(default_factory=list)
    copy_solution: bool = False  # by default for performance reasons

    def __init__(self, methods: List[object] = None, copy_solution: bool = False):
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
    instance: Instance
    root: object = None
    objective_value = None
    best_solution : Solution = None
    all_solution : List[Solution] = field(default_factory=list)
    start_time : float = 0
    runtime : float = 0

    @dataclass
    class Node():
        lower_bound: float = None
        if_solution: bool = False
        partial_solution: object = None
        sub_nodes: List[object] = field(default_factory=list)

        def delete(self):
            """To delete the variable definitely
            """
            for node in self.sub_nodes:
                node.delete()
            del self

    def branch(self, node: Node):
        """branching strategy, to be redefined

        Args:
            node (Node): node to branch from
        """
        pass

    def bound(self, node: Node):
        """bounding method, to be redefined

        Args:
            node (Node): node to bound
        """
        pass

    def discard(self, root: Node, best_solution: float, objective: Objective):
        """prunes the search tree sections where we are certain a better solution will not be find there

        Args:
            root (Node): root node of the tree
            best_solution (float): best objective value
            objective (Objective): objective to be considered, to know if it's a minimization or a maximization problem
        """
        if root.lower_bound is not None:
            if objective.value > 0 and root.lower_bound < best_solution : root = None
            elif objective.value < 0 and root.lower_bound > best_solution : root = None
        #for node in root.sub_nodes :
        #    if objective.value > 0 and node.lower_bound < best_solution : node = None
        #    elif objective.value < 0 and node.lower_bound > best_solution : node = None

    def objective(self, node : Node):
        """objective value evaluator, to be redefined

        Args:
            node (Node): node to be evaluated as a complete solution
        """
        pass

    def solution_format(self, partial_solution : object, objective_value) :
        pass
    
    def solve(self, root : Node = None):
        """recursive function to perform Branch&Bound on the instance attribute

        Args:
            root (Node, optional): starting node. Defaults to None.
        """
        if root is None:
            root = self.Node()
            self.root = root
            self.start_time = perf_counter()
        self.branch(root) 
        if root.sub_nodes[0].if_solution is False :
            for node in root.sub_nodes: self.bound(node)
            sorted_sub_nodes = root.sub_nodes
            sorted_sub_nodes.sort(reverse= self.instance.get_objective().value > 0, key = lambda node : node.lower_bound)
            for node in sorted_sub_nodes :
                if self.best_solution is not None : 
                    if self.instance.get_objective().value > 0 and node.lower_bound < self.objective_value : node = None
                    elif self.instance.get_objective().value < 0 and node.lower_bound > self.objective_value : node = None
                if node is not None : self.solve(node)
        else :
            for node in root.sub_nodes: 
                node.lower_bound = self.objective(node)
                solution = self.solution_format(node.partial_solution,node.lower_bound)
                self.all_solution.append(solution)
                if self.best_solution is None or (self.instance.get_objective().value > 0 and self.objective_value < node.lower_bound) :
                    self.objective_value = node.lower_bound
                    self.best_solution = solution
                elif self.best_solution is None or (self.instance.get_objective().value < 0 and node.lower_bound < self.objective_value) :
                    self.objective_value = node.lower_bound
                    self.best_solution = solution
        self.runtime = perf_counter() - self.start_time
                
    def get_solve_result(self):
        return SolveResult(best_solution=self.best_solution,status=SolveStatus.OPTIMAL,runtime=self.runtime,solutions=self.all_solution)   

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
