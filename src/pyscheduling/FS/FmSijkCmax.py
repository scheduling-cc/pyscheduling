from dataclasses import dataclass, field
from pathlib import Path
from random import randint, uniform
from typing import List

import pyscheduling.FS.FlowShop as FlowShop
import pyscheduling.FS.FS_methods as FS_methods
import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Solver


@dataclass
class FmSijkCmax_Instance(FlowShop.FlowShopInstance):
    P: List[List[int]] = field(default_factory=list)  # Processing time
    S: List[List[List[int]]] = field(default_factory=list) # Setup time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            FmSijkCmax_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of configuration
        m = int(ligne0[2])  # number of jobs
        i = 2
        instance = cls("test", n, m)
        instance.P, i = instance.read_P(content, i)
        instance.S, i = instance.read_S(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, n: int, m: int, protocol: FlowShop.GenerationProtocol = FlowShop.GenerationProtocol.BASE, law: FlowShop.GenerationLaw = FlowShop.GenerationLaw.UNIFORM, Pmin: int = -1, Pmax: int = -1, Gamma: float = 0.0, Smin:  int = -1, Smax: int = -1, InstanceName: str = ""):
        """Random generation of FmSijkCmax problem instance

        Args:
            n (int): number of jobs of the instance
            m (int): number of machines of the instance
            protocol (FlowShop.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to FlowShop.GenerationProtocol.VALLADA.
            law (FlowShop.GenerationLaw, optional): probablistic law of generation. Defaults to FlowShop.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            Gamma (float, optional): Setup time factor. Defaults to 0.0.
            Smin (int, optional): Minimal setup time. Defaults to -1.
            Smax (int, optional): Maximal setup time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            FmSijkCmax_Instance: the randomly generated instance
        """
        if(Pmin == -1):
            Pmin = randint(1, 100)
        if(Pmax == -1):
            Pmax = randint(Pmin, 100)
        if(Gamma == 0.0):
            Gamma = round(uniform(1.0, 3.0), 1)
        if(Smin == -1):
            Smin = randint(1, 100)
        if(Smax == -1):
            Smax = randint(Smin, 100)
        instance = cls(InstanceName, n, m)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.S = instance.generate_S(
            protocol, law, instance.P, Gamma, Smin, Smax)
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+"  "+str(self.m)+"\n")
        f.write(str(self.m)+"\n")
        for i in range(self.n):
            for j in range(self.m):
                f.write("\t"+str(j)+"\t"+str(self.P[i][j]))
            f.write("\n")
        f.write("SSD\n")
        for i in range(self.m):
            f.write("M"+str(i)+"\n")
            for j in range(self.n):
                for k in range(self.n):
                    f.write(str(self.S[i][j][k])+"\t")
                f.write("\n")
        f.close()

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.MINIT

    def get_objective(self):
        """to get the objective tackled by the instance

        Returns:
            RootProblem.Objective: Makespan
        """
        return RootProblem.Objective.Cmax


class Heuristics(FS_methods.Heuristics):

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

class Metaheuristics(FS_methods.Metaheuristics):
    pass