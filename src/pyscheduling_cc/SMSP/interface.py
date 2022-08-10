from enum import Enum
from pathlib import Path
from collections import Counter


import pyscheduling_cc.Problem as Problem
import pyscheduling_cc.SMSP as sm

class Constraints(Enum):
    W = "weight"
    R = "release"
    S = "setup"
    D = "due"

    @classmethod
    def toString(cls):
        return cls.W.value + "\n" + cls.R.value + "\n" + cls.S.value + "\n" + cls.D.value


class problem():
    instance : sm.SingleMachine.SingleInstance
    constraints : list[Constraints]
    objective : Problem.Objective
    heuristics = None
    metaheuristics = None

    def __init__(self) -> None:
        self.instance = None
        self.constraints = []
        self.objective = None

    def generate_random(self, **data):
        if Counter(self.constraints) == Counter([Constraints.W]) and self.objective == Problem.Objective.wiCi: 
            self.instance = sm.wiCi.wiCi_Instance.generate_random(**data)
            heuristics = sm.wiCi.Heuristics.all_methods()
            metaheuristics = sm.wiCi.Metaheuristics.all_methods()
        elif Counter(self.constraints) == Counter([Constraints.W, Constraints.R]) and self.objective == Problem.Objective.wiCi: 
            self.instance = sm.riwiCi.riwiCi_Instance.generate_random(**data)
            heuristics = sm.riwiCi.Heuristics.all_methods()
            metaheuristics = sm.riwiCi.Metaheuristics.all_methods()
        elif Counter(self.constraints) == Counter([Constraints.D, Constraints.W]) and self.objective == Problem.Objective.wiTi: 
            self.instance = sm.wiTi.wiTi_Instance.generate_random(**data)
            heuristics = sm.wiTi.Heuristics.all_methods()
            metaheuristics = sm.wiTi.Metaheuristics.all_methods()
        elif Counter(self.constraints) == Counter([Constraints.D, Constraints.W, Constraints.R]) and self.objective == Problem.Objective.wiTi: 
            self.instance = sm.riwiTi.riwiTi_Instance.generate_random(**data)
            heuristics = sm.riwiTi.Heuristics.all_methods()
            metaheuristics = sm.riwiTi.Metaheuristics.all_methods()
        elif Counter(self.constraints) == Counter([Constraints.D, Constraints.W, Constraints.S]) and self.objective == Problem.Objective.wiTi: 
            self.instance = sm.sijwiTi.sijwiTi_Instance.generate_random(**data)
            heuristics = sm.sijwiTi.Heuristics.all_methods()
            metaheuristics = sm.sijwiTi.Metaheuristics.all_methods()
        elif Counter(self.constraints) == Counter([Constraints.D, Constraints.W, Constraints.R ,Constraints.S]) and self.objective == Problem.Objective.wiTi: 
            self.instance = sm.risijwiTi.risijwiTi_Instance.generate_random(**data)
            heuristics = sm.risijwiTi.Heuristics.all_methods()
            metaheuristics = sm.risijwiTi.Metaheuristics.all_methods()
        elif Counter(self.constraints) == Counter([Constraints.W, Constraints.S]) and self.objective == Problem.Objective.Cmax: 
            self.instance = sm.sijCmax.sijCmax_Instance.generate_random(**data)
            heuristics = sm.sijCmax.Heuristics.all_methods()
            metaheuristics = sm.sijCmax.Metaheuristics.all_methods()
        elif Counter(self.constraints) == Counter([Constraints.W, Constraints.R, Constraints.S]) and self.objective == Problem.Objective.Cmax: 
            self.instance = sm.risijCmax.risijCmax_Instance.generate_random(**data)
            heuristics = sm.risijCmax.Heuristics.all_methods()
            metaheuristics = sm.risijCmax.Metaheuristics.all_methods()
        else : 
            if self.objective is None : raise TypeError("Please, initialize the objective")
            else: raise TypeError("This configuration of constraints and objective is not offered in pyscheduling yet")
        

        self.heuristics = dict(zip([func.__name__ for func in heuristics], heuristics))
        self.metaheuristics = dict(zip([func.__name__ for func in metaheuristics], metaheuristics))

    def setConstraints(self, constraints : list[Constraints]):
        for constraint in constraints :
            if type(constraint) != Constraints : raise  TypeError("Only Constraints Enum elements are allowed :\n"+Constraints.toString())
        self.constraints = constraints

    def addConstraints(self, constraints):
        if type(constraints) == list : 
            for constraint in constraints :
                if type(constraint) != Constraints : raise  TypeError("Only Constraints Enum elements are allowed :\n"+Constraints.toString())
            self.constraints.extend([constraint for constraint in constraints if constraint not in self.constraints])
        elif (type(constraints) == Constraints) and (constraints not in self.constraints): self.constraints.append(constraints)
        else: raise  TypeError("Only Constraints Enum elements are allowed :\n"+Constraints.toString())

    def removeConstraint(self, constraint : Constraints):
        if type(constraint) == Constraints : self.constraints.remove(constraint)
        else : raise  TypeError("Argument to remove is not a constraint")

    def setObjective(self, objective : Problem.Objective):
        if type(objective) != Problem.Objective : raise TypeError("objective must be an Objective Enum element :\n"+Problem.Objective.toString())
        else :
            if(objective == Problem.Objective.wiTi) and Constraints.D not in self.constraints:
                raise TypeError("Due dates must be added as a constraint in order to have Lateness as an objective")
            else : self.objective = objective

    def solve(self, method : object, **data):
        if not callable(method):
            raise ValueError("Argument passed to solve method is not a function")
        else:
            try:
                return method(self.instance, **data)
            except:
                print("Do correctly use the method as explained below :\n" +
                        method.__doc__)