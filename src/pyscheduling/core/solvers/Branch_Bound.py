from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import queue
from time import perf_counter

from pyscheduling.Problem import BaseInstance, BaseSolution, SolveStatus, SolveResult


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

    def objective(self, node : Node):
        """objective value evaluator, to be redefined

        Args:
            node (Node): node to be evaluated as a complete solution
        """
        pass

    def solution_format(self, partial_solution : object, objective_value) :
        pass
    
    def solve(self, lower_bound : callable, minimize = True, root : Node = None, max_time = float('inf'), upper_bound = float('+inf')):
        reverse = 1 if minimize else -1
        root = self.Node()
        self.root = root
        self.start_time = perf_counter()
        Q = queue.LifoQueue()
        Q.put(root)
        while not Q.empty() and (self.objective_value is None or perf_counter()-self.start_time <= max_time):
            node = Q.get()
            self.branch(node)
            for sub_node in sorted(node.sub_nodes, key= lambda x: lower_bound(self,x) if not x.if_solution else self.objective(x), reverse=minimize):
                if sub_node.if_solution is True:
                    solution = self.solution_format(sub_node.partial_solution, sub_node.lower_bound)
                    if self.best_solution is None or (reverse * self.objective_value >= reverse * sub_node.lower_bound):
                        self.objective_value = sub_node.lower_bound
                        self.best_solution = solution
                else:
                    if self.best_solution is not None and ((reverse * self.objective_value < reverse * sub_node.lower_bound) or (reverse * upper_bound <= reverse * sub_node.lower_bound)):
                        continue
                    else: 
                        Q.put(sub_node)
        self.runtime = perf_counter() - self.start_time
                
    def get_solve_result(self):
        return SolveResult(best_solution=self.best_solution,status=SolveStatus.OPTIMAL,runtime=self.runtime,solutions=self.all_solution)   