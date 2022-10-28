from crypt import methods
from time import perf_counter

import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Constraints, Objective
import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.SMSP.SingleMachine import single_instance
import pyscheduling.SMSP.SM_Methods as Methods
from pyscheduling.SMSP.SM_Methods import ExactSolvers


@single_instance([Constraints.W, Constraints.R, Constraints.S], Objective.wiCi)
class risijwiCi_Instance(SingleMachine.SingleInstance):

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.constructive


class Heuristics(Methods.Heuristics):
    
    @staticmethod
    def list_heuristic(instance: risijwiCi_Instance, rule_number: int = 0) -> RootProblem.SolveResult:
        """contains a list of static dispatching rules to be chosen from

        Args:
            instance (riwiCi_Instance): Instance to be solved
            rule_number (int, optional) : Index of the rule to use. Defaults to 1.

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        if rule_number==1: # Increasing order of the release time
            sorting_func = lambda instance, job_id : instance.R[job_id]
            reverse = False
        elif rule_number==2: # WSPT
            sorting_func = lambda instance, job_id : float(instance.W[job_id])/float(instance.P[job_id])
            reverse = True
        elif rule_number==3: #WSPT including release time in the processing time
            sorting_func = lambda instance, job_id : float(instance.W[job_id])/float(instance.R[job_id]+instance.P[job_id])
            reverse = True

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

