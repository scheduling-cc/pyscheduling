from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


from pyscheduling_cc.Problem import Objective
from pyscheduling_cc.PMSP import *

class Constraints(Enum):
    W = "weight"
    R = "release"
    S = "setup"
    D = "due"

    @classmethod
    def toString(cls):
        return cls.W.value + "\n" + cls.R.value + "\n" + cls.S.value + "\n" + cls.D.value

    def __lt__(self,other):
        return self.name < other.name

problems = {
    ((Constraints.S,),Objective.Cmax) : (RmSijkCmax.RmSijkCmax_Instance,RmSijkCmax.Heuristics,RmSijkCmax.Metaheuristics), 
    ((Constraints.R, Constraints.S),Objective.Cmax) : (RmriSijkCmax.RmriSijkCmax_Instance,RmriSijkCmax.Heuristics,RmriSijkCmax.Metaheuristics)
}

@dataclass
class Problem():
    key = None
    instance : ParallelMachines.ParallelInstance
    constraints : list[Constraints]
    objective : Objective
    heuristics = None
    metaheuristics = None

    def __init__(self) -> None:
        self.instance = None
        self.constraints = []
        self.objective = None

    def set_key(self):
        if self.objective is not None and self.constraints != []: 
            self.constraints.sort()
            self.key = (tuple(self.constraints),self.objective)
        else : self.key = None

    def generate_random(self, **data):
        if self.key is not None:
            instance_class, heuristics_class, metaheuristics_class = problems[self.key]

            self.instance = instance_class.generate_random(**data)
            
            heuristics = heuristics_class.all_methods()
            self.heuristics = dict(zip([func.__name__ for func in heuristics], heuristics))
            metaheuristics = metaheuristics_class.all_methods()
            self.metaheuristics = dict(zip([func.__name__ for func in metaheuristics], metaheuristics))
        else: raise TypeError("Please add constraints or set objective")
    
    def read_txt(self, path : Path):
        if self.key is not None:
            instance_class, heuristics_class, metaheuristics_class = problems[self.key]
            
            self.instance = instance_class.read_txt(path)

            heuristics = heuristics_class.all_methods()
            self.heuristics = dict(zip([func.__name__ for func in heuristics], heuristics))
            metaheuristics = metaheuristics_class.all_methods()
            self.metaheuristics = dict(zip([func.__name__ for func in metaheuristics], metaheuristics))
        else: raise TypeError("Please add constraints or set objective")
    
    def add_constraints(self, constraints):
        if type(constraints) == list : 
            for constraint in constraints :
                if type(constraint) != Constraints : raise  TypeError("Only Constraints Enum elements are allowed :\n"+Constraints.toString())
            self.constraints.extend([constraint for constraint in constraints if constraint not in self.constraints])
        elif (type(constraints) == Constraints) and (constraints not in self.constraints): self.constraints.append(constraints)
        else: raise  TypeError("Only Constraints Enum elements are allowed :\n"+Constraints.toString())
        self.set_key()

    def remove_constraint(self, constraint : Constraints):
        if type(constraint) == Constraints : self.constraints.remove(constraint)
        else : raise  TypeError("Argument to remove is not a constraint")
        self.set_key()

    def set_objective(self, objective : Objective):
        if type(objective) != Objective : raise TypeError("objective must be an Objective Enum element :\n"+Objective.toString())
        else :
            if(objective == Objective.wiTi) and Constraints.D not in self.constraints:
                raise TypeError("Due dates must be added as a constraint in order to have Lateness as an objective")
            else : self.objective = objective
        self.set_key()

    def solve(self, method : object, **data):
        if not callable(method):
            raise ValueError("Argument passed to solve method is not a function")
        else:
            try:
                return method(self.instance, **data)
            except:
                raise TypeError("Do correctly use the method as explained below :\n" +
                        method.__doc__)