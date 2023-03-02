from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pyscheduling.JS.JobShop as JobShop
import pyscheduling.JS.JS_methods as js_methods
import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import GenerationLaw, Solver


@dataclass
class JmriSijkwiCi_Instance(JobShop.JobShopInstance):
    P: List[List[int]] = field(default_factory=list)  # Processing time
    W: List[int] = field(default_factory=list)
    R: List[int] = field(default_factory=list)
    S: List[List[List[int]]] = field(default_factory=list) # Setup time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            JmriSijkwiCi_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of configuration
        m = int(ligne0[1])  # number of jobs
        i = 1
        instance = cls("test", n, m)
        instance.P, i = instance.read_P(content, i)
        instance.W, i = instance.read_1D(content, i)
        instance.R, i = instance.read_1D(content, i)
        instance.S, i = instance.read_S(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, n: int, m: int, instance_name: str = "",
                        protocol: JobShop.GenerationProtocol = JobShop.GenerationProtocol.BASE, law: GenerationLaw = GenerationLaw.UNIFORM,
                        Pmin: int = 10, Pmax: int = 100,
                        Wmin: int = 1, Wmax: int = 1,
                        alpha: float = 2.0,
                        Gamma: float = 2.0, Smin: int = 1, Smax: int = 50):
        """Random generation of JmriSijkwiCi problem instance
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
            JmriSijkwiCi_Instance: the randomly generated instance
        """
        instance = cls(instance_name, n, m)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.W = instance.generate_W(protocol, law, Wmin, Wmax)
        instance.R = instance.generate_R(
                protocol, law, instance.P, Pmin, Pmax, alpha)
        instance.S = instance.generate_S(
            protocol, law, instance.P, Gamma, Smin, Smax)
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+" "+str(self.m)+"\n")

        f.write("Processing times\n")
        for job in self.P:
            for operation in job:
                f.write(str(operation[0])+"\t"+str(operation[1])+"\t")
            f.write("\n")

        f.write("Weights\n")
        for i in range(self.n):
            f.write(str(self.W[i])+"\t")

        f.write("\nRelease time\n")
        for i in range(self.n):
            f.write(str(self.R[i])+"\t")

        f.write("\nSSD\n")
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
        return Heuristics.BIBA

    def get_objective(self):
        """to get the objective tackled by the instance

        Returns:
            RootProblem.Objective:
        """
        return RootProblem.Objective.wiCi

class Heuristics(js_methods.Heuristics):

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]