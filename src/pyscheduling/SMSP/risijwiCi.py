from dataclasses import dataclass
from math import exp
from typing import ClassVar, List

import pyscheduling.Problem as Problem
import pyscheduling.SMSP.SingleMachine as SingleMachine
import pyscheduling.SMSP.SM_methods as Methods
from pyscheduling.Problem import Objective
from pyscheduling.SMSP.SingleMachine import Constraints
from pyscheduling.SMSP.SM_methods import ExactSolvers


@dataclass(init=False)
class risijwiCi_Instance(SingleMachine.SingleInstance):

    P: List[int]
    W: List[int]
    R: List[int]
    D: List[int]
    S: List[List[int]]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.W, Constraints.R, Constraints.S]
    objective: ClassVar[Objective] = Objective.wiCi

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.BIBA


class Heuristics(Methods.Heuristics):
    
    @staticmethod
    def list_heuristic(instance: risijwiCi_Instance, rule_number: int = 0, reverse = False) -> Problem.SolveResult:
        """contains a list of static dispatching rules to be chosen from

        Args:
            instance (riwiCi_Instance): Instance to be solved
            rule_number (int, optional) : Index of the rule to use. Defaults to 1.

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        s_bar = sum(sum(instance.S[l]) for l in range(instance.n) ) / (instance.n * instance.n)
        default_rule = lambda instance, job_id : instance.R[job_id]
        rules_dict = {
            0: default_rule,
            1: lambda instance, job_id : instance.W[job_id] / instance.P[job_id],
            2: lambda instance, job_id : instance.W[job_id] / (instance.R[job_id]+instance.P[job_id]),
            3: lambda instance, job_id : exp(-(sum(instance.S[l][job_id] for l in range(instance.n))) / ( 0.2 * s_bar) ) * instance.W[job_id] / (instance.R[job_id]+instance.P[job_id])
        }
        
        sorting_func = rules_dict.get(rule_number, default_rule)

        return Methods.Heuristics.dispatch_heuristic(instance, sorting_func, reverse)


    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

class Metaheuristics(Methods.Metaheuristics):
    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

