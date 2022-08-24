from math import exp
import sys
from dataclasses import dataclass, field
from random import randint, uniform
from pathlib import Path
from time import perf_counter

from matplotlib import pyplot as plt

import pyscheduling_cc.Problem as RootProblem
from pyscheduling_cc.Problem import Solver
import pyscheduling_cc.SMSP.SingleMachine as SingleMachine
import pyscheduling_cc.SMSP.SM_Methods as Methods
from pyscheduling_cc.SMSP.SM_Methods import ExactSolvers


@dataclass
class riPrecLmax_Instance(SingleMachine.SingleInstance):
    P: list[int] = field(default_factory=list)  # Processing time
    R: list[int] = field(default_factory=list) # release time
    D: list[int] = field(default_factory=list) # due time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            riPrecLmax_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of jobs
        i = 1
        instance = cls("test", n)
        instance.P, i = instance.read_P(content, i)
        instance.R, i = instance.read_R(content, i)
        instance.D, i = instance.read_D(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int,  protocol: SingleMachine.GenerationProtocol = SingleMachine.GenerationProtocol.BASE, law: SingleMachine.GenerationLaw = SingleMachine.GenerationLaw.UNIFORM, Pmin: int = 1, Pmax: int = -1, alpha : float = 0.0, due_time_factor : float = 0.0, InstanceName: str = ""):
        """Random generation of RmSijkCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            protocol (SingleMachine.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to SingleMachine.GenerationProtocol.VALLADA.
            law (SingleMachine.GenerationLaw, optional): probablistic law of generation. Defaults to SingleMachine.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            riPrecLmax_Instance: the randomly generated instance
        """
        if(Pmax == -1):
            Pmax = Pmin + randint(1, 100)
        if(alpha == 0.0):
            alpha = round(uniform(1.0, 3.0), 1)
        if(due_time_factor == 0.0):
            due_time_factor = round(uniform(0, 1), 1)
        instance = cls(InstanceName, jobs_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.R = instance.generate_R(protocol,law,instance.P,Pmin,Pmax,alpha)
        instance.D = instance.generate_D(protocol,law,instance.P,Pmin,Pmax,due_time_factor,RJobs=instance.R)
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n))
        f.write("\nProcessing time\n")
        for i in range(self.n):
            f.write(str(self.P[i])+"\t")
        f.write("\nRelease time\n")
        for i in range(self.n):
            f.write(str(self.R[i])+"\t")
        f.write("\nDue time\n")
        for i in range(self.n):
            f.write(str(self.D[i])+"\t")
        f.close()


    def get_objective(self):
        return RootProblem.Objective.wiTi

    def init_sol_method(self):
        #return Heuristics.ACT_WSECi
        pass


class Heuristics():

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

class BB(RootProblem.Branch_Bound):
    def branch(self, node : RootProblem.Branch_Bound.Node):
        pass
    def bound(self, node : RootProblem.Branch_Bound.Node):
        pass
    def objective(self, node : RootProblem.Branch_Bound.Node):
        pass