from dataclasses import dataclass
from pathlib import Path


from pyscheduling.Problem import Objective, Constraints
from pyscheduling.SMSP import *


problems = {
    ((Constraints.W,),Objective.wiCi) : (wiCi.wiCi_Instance,wiCi.Heuristics,wiCi.Metaheuristics),
    ((Constraints.R, Constraints.W),Objective.wiCi) : (riwiCi.riwiCi_Instance,riwiCi.Heuristics,riwiCi.Metaheuristics),
    ((Constraints.D, Constraints.W),Objective.wiTi) : (wiTi.wiTi_Instance,wiTi.Heuristics,wiTi.Metaheuristics),
    ((Constraints.D, Constraints.R, Constraints.W),Objective.wiTi) : (riwiTi.riwiTi_Instance,riwiTi.Heuristics,riwiTi.Metaheuristics), 
    ((Constraints.D, Constraints.S, Constraints.W),Objective.wiTi) : (sijwiTi.sijwiTi_Instance,sijwiTi.Heuristics,sijwiTi.Metaheuristics),
    ((Constraints.D, Constraints.R ,Constraints.S, Constraints.W),Objective.wiTi) : (risijwiTi.risijwiTi_Instance,risijwiTi.Heuristics,risijwiTi.Metaheuristics), 
    ((Constraints.S, Constraints.W),Objective.Cmax) : (sijCmax.sijCmax_Instance,sijCmax.Heuristics,sijCmax.Metaheuristics), 
    ((Constraints.R, Constraints.S, Constraints.W),Objective.Cmax) : (risijCmax.risijCmax_Instance,risijCmax.Heuristics,risijCmax.Metaheuristics)
}

@dataclass
class Problem():
    key = None
    instance : SingleMachine.SingleInstance
    constraints : list[Constraints]
    objective : Objective
    heuristics = None
    metaheuristics = None

    def __init__(self) -> None:
        self.instance = None
        self.constraints = []
        self.objective = None

    def set_key(self):
        """set the key attribute of the problem instance based on objective and constraints attributes
         as described in problems global dict
        """
        if self.objective is not None and self.constraints != []: 
            self.constraints.sort()
            self.key = (tuple(self.constraints),self.objective)
            try:
                problems[self.key]
            except:
                raise KeyError("The combination of constraints and objective you have entered is not yet handled by pyscheduling, check the documentation for more.\n")
        else : self.key = None

    def generate_random(self, **data):
        """Generate randomly the instance attribute along with its corresponding heuristics and metaheuristics

        Raises:
            TypeError: In case the key is None, it means the user didnt add constraints or an objective
        """
        if self.key is not None:
            instance_class, heuristics_class, metaheuristics_class = problems[self.key]

            self.instance = instance_class.generate_random(**data)
            
            heuristics = heuristics_class.all_methods()
            self.heuristics = dict(zip([func.__name__ for func in heuristics], heuristics))
            metaheuristics = metaheuristics_class.all_methods()
            self.metaheuristics = dict(zip([func.__name__ for func in metaheuristics], metaheuristics))
        else: raise TypeError("Please add constraints or set objective")
    
    def read_txt(self, path : Path):
        """Read the instance attribute from a text file corresponding to the right input format of the instance type

        Args:
            path (Path): path of the instance file to be read

        Raises:
            TypeError: In case the key is None, it means the user didnt add constraints or an objective
        """
        if self.key is not None:
            instance_class, heuristics_class, metaheuristics_class = problems[self.key]
            
            self.instance = instance_class.read_txt(path)

            heuristics = heuristics_class.all_methods()
            self.heuristics = dict(zip([func.__name__ for func in heuristics], heuristics))
            metaheuristics = metaheuristics_class.all_methods()
            self.metaheuristics = dict(zip([func.__name__ for func in metaheuristics], metaheuristics))
        else: raise TypeError("Please add constraints or set objective")
    
    def add_constraints(self, constraints):
        """Adds constraints to the attribute constraints

        Args:
            constraints (object): can be a single Constraints type object or a list of Constraints type objects 

        Raises:
            TypeError: If one of the constraints list element is not a Constraints type object
            TypeError: If the constraints object is not a Constraints type object
        """
        if type(constraints) == list : 
            for constraint in constraints :
                if type(constraint) != Constraints : raise  TypeError("Only Constraints Enum elements are allowed :\n"+Constraints.toString())
            self.constraints.extend([constraint for constraint in constraints if constraint not in self.constraints])
        elif (type(constraints) == Constraints) and (constraints not in self.constraints): self.constraints.append(constraints)
        else: raise  TypeError("Only Constraints Enum elements are allowed :\n"+Constraints.toString())
        self.set_key()

    def remove_constraint(self, constraint : Constraints):
        """to remove a constraint from constraints attribute

        Args:
            constraint (Constraints): constraint to be removed

        Raises:
            TypeError: In case the argument is not of Constraints type
        """
        if type(constraint) == Constraints : self.constraints.remove(constraint)
        else : raise  TypeError("Argument to remove is not a constraint")
        self.set_key()

    def set_objective(self, objective : Objective):
        """set the objective attribute

        Args:
            objective (Objective): chosen objective

        Raises:
            TypeError: In case the argument passed is not of Objective type
            TypeError: In case Lateness as wanted objective but due dates are not in constraints attribute
        """
        if type(objective) != Objective : raise TypeError("objective must be an Objective Enum element :\n"+Objective.toString())
        else :
            if(objective == Objective.wiTi) and Constraints.D not in self.constraints:
                raise TypeError("Due dates must be added as a constraint in order to have Lateness as an objective")
            else : self.objective = objective
        self.set_key()

    def solve(self, method : object, **data):
        """call the method passed as an argument to solve the instance attribute

        Args:
            method (object): callable method to solve the instance attribute, can be in the heuristics or metaheuristics list or an user-created method

        Raises:
            ValueError: The method argument is not a callable
            TypeError: In case the method is not properly used

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        if not callable(method):
            raise ValueError("Argument passed to solve method is not a function")
        else:
            try:
                return method(self.instance, **data)
            except:
                raise TypeError("Do correctly use the method as explained below :\n" +
                        method.__doc__)