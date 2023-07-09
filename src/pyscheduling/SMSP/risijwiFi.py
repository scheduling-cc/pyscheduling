from dataclasses import dataclass, field
from math import exp
from typing import ClassVar, List

import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.core.base_solvers import BaseSolver
from pyscheduling.Problem import Objective
from pyscheduling.SMSP.SingleMachine import Constraints
from pyscheduling.SMSP.solvers import BIBA, DispatchHeuristic


@dataclass(init=False)
class risijwiFi_Instance(SingleMachine.SingleInstance):

    P: List[int]
    W: List[int]
    R: List[int]
    D: List[int]
    S: List[List[int]]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.W, Constraints.R, Constraints.S]
    objective: ClassVar[Objective] = Objective.wiFi
    init_sol_method: BaseSolver = BIBA()


@dataclass
class ListHeuristic(BaseSolver):

    rule_number: int = 1
    reverse : bool = False

    def solve(self, instance: risijwiFi_Instance):
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
        
        sorting_func = rules_dict.get(self.rule_number, default_rule)

        return DispatchHeuristic(rule=sorting_func, reverse=self.reverse).solve(instance)
