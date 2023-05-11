from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import plotly.figure_factory as ff

Job = namedtuple('Job', ['id', 'start_time', 'end_time'])

class GenerationProtocol(Enum):
    BASE = 1

class RandomDistrib(Enum):
    UNIFORM = 1
    NORMAL = 2

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


@dataclass
class BaseInstance(ABC):

    name: str
    n: int

    def __init__(self, n: int, m: int = 1,
                name: str = "Unknown", **kwargs):
        
        self.constraints = sorted(self.constraints, key= lambda x: x._value)
        self.n = n
        self.m = m
        self.name = name
        
        for constraint in self.constraints:
            init_value = kwargs.get(constraint._name, None)
            constraint.create(self,init_value)

    @classmethod
    def generate_random(cls, n: int, m: int = 1, name: str = "Unknown",
                        protocol: GenerationProtocol = GenerationProtocol.BASE,
                        law: RandomDistrib = RandomDistrib.UNIFORM,
                        Wmin: int = 1, Wmax: int = 4,
                        Pmin: int = 10, Pmax: int = 50,
                        alpha: float = 0.8,
                        due_time_factor: float = 0.8,
                        gamma: float = 0.5):
        
        """Random generation of a problem instance
        
        Args:
            n (int): number of jobs of the instance
            m (int): number of machines of the instance
            instance_name (str, optional): name to give to the instance. Defaults to "Unknown".
            protocol (GenerationProtocol, optional): given protocol of generation of random instances. Defaults to GenerationProtocol.BASE.
            law (FlowShop.GenerationLaw, optional): probablistic law of generation. Defaults to GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to 10.
            Pmax (int, optional): Maximal processing time. Defaults to 50.
            alpha (float, optional): Release time factor. Defaults to 0.8.
            due_time_factor (float, optional): Due time factor. Defaults to 0.8.
            gamma (float, optional): Setup time factor. Defaults to 0.5.
            
        Returns:
            BaseInstance: the randomly generated instance
        """

        instance = cls(n, m = m, name=name)
        
        args_dict = {   "protocol": protocol, "law":law,
                        "Wmin":Wmin, "Wmax":Wmax,
                        "Pmin":Pmin, "Pmax":Pmax,
                        "alpha":alpha,
                        "due_time_factor":due_time_factor,
                        "gamma":gamma}
        
        for constraint in instance.constraints:
            constraint.generate_random(instance,**args_dict)

        return instance 
    
    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format
        
        Args:
            path (Path): path to the txt file of type Path from the pathlib module
        
        Raises:
            FileNotFoundError: when the file does not exist
        
        Returns:
            BaseInstance: the read instance
        """
        path = Path(str(path))
        with open(path, "r") as f:
            content = f.read().split('\n')
            ligne0 = content[0].split(' ')
            n = int(ligne0[0])  # number of jobs
            m = int(ligne0[2]) if len(ligne0) > 2 else 1  # number of machines
            i = 1
            instance = cls(n, m = m, name = path.name)
            for constraint in instance.constraints:
                i = constraint.read(instance,content,i)

        return instance

    def to_txt(self, path : Path):
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        with open(path, "w") as f:
            f.write(str(self.n)+"  "+str(self.m)+"\n")
            f.write(str(self.m)+"\n")
            for constraint in self.constraints:
                constraint.write(self,f)

    def get_objective(self):
        """getter to the objective class attribute
        
        Returns:
            Objective: the objective of the problem
        """
        return self.objective


@dataclass
class BaseSolution(ABC):

    instance: BaseInstance
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
    
    def _plot_tasks(self, tasks_df: List[dict], path: Path = None):
        """Plots the tasks (in plotly dict format), it can be called by all problems.

        Args:
            tasks_df (List[dict]): Tasks list of dicts specifying start and end dates and description
            path (Path, optional): The path to export the diagram, if not specified it is not exported but shown inline. Defaults to None.
        """
        barwidth = 0.2
        colors = {'Idle': 'rgb(238, 238, 238)',
                'Setup': 'rgb(255, 178, 0)',
                'Processing': 'rgb(39, 123, 192)',
                'black': 'rgb(0, 0, 0)'}
        
        cmax_value = max(task["Finish"] for task in tasks_df)
        y_max = self.instance.m if hasattr(self.instance, "m") else 0.5

        fig = ff.create_gantt(tasks_df, colors=colors, index_col='Type', show_colorbar=True,
                            group_tasks=True, showgrid_x=True, showgrid_y=True, bar_width=barwidth)

        fig.update_yaxes(autorange="reversed") #if not specified as 'reversed', the tasks will be listed from bottom up       
        fig.update_layout(
            xaxis_type='linear',
            title='<b>Gantt Chart</b>',
            #bargap=0.1,
            #width=850,
            #height=500,              
            xaxis_title="<b>Time</b>", 
            yaxis_title="<b>Machine</b>",
            font=dict(
                size=16,
            ),
            hovermode = "x"
        )
        #fig.add_vline(x=cmax_value, line_width=4, line_dash="dashdot", line_color="Red")

        # Add rectangle shapes
        if not hasattr(self.instance, "S"):
            for task in tasks_df:
                if task["Type"] == "Processing":
                    y_ref = int(task["Task"][1:])
                    fig.add_shape(type="rect", x0=task["Start"], x1=task["Finish"], y0=y_ref-barwidth, y1=y_ref+barwidth, line=dict(color=colors["black"])) 
        
        # add annotations
        annots = []
        for task in tasks_df:
            if task["Type"] == "Processing":
                machine = task["Task"]
                x_annot = task["Start"] + (task["Finish"] - task["Start"] + 1) // 2
                y_annot = y_max - (int(machine[1:]) + 1)
                annots.append( dict(x= x_annot,y=y_annot,text=task["Description"], showarrow=False, font=dict(color='white')) )

        #print(annots)

        # plot figure
        fig['layout']['annotations'] = annots

        # Cmax value
        fig.add_annotation(x=cmax_value, y=-2*barwidth,
            text=f'Objective_value: {self.objective_value}',
            font=dict(size=12, color="red", family="Courier New, monospace"), align="right"
        )

        if path is not None:
            fig.write_image(path)
        else:
            print("Showing")
            fig.show()

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

    all_solutions: List[BaseSolution]
    best_solution: BaseSolution  # Needs to be consistent with "all_solutions" list
    time_to_best: float
    solve_status: SolveStatus
    runtime: float
    kpis: Dict[str, object]  # Other metrics that are problem / solver specific

    def __init__(self, best_solution: BaseSolution = None, runtime: float = -1,
                 time_to_best: float = -1, status: SolveStatus = SolveStatus.FEASIBLE,
                 solutions: List[BaseSolution] = None, kpis: Dict[str, object] = None):
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

    def improve(self, solution: BaseSolution) -> BaseSolution:
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
    instance: BaseInstance
    root: object = None
    objective_value = None
    best_solution : BaseSolution = None
    all_solution : List[BaseSolution] = field(default_factory=list)
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

    def solve(self, instance: BaseInstance, **data) -> SolveResult:
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
