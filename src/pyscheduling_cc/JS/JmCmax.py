from dataclasses import dataclass, field
from pathlib import Path
from random import randint

from time import perf_counter


import pyscheduling_cc.Problem as RootProblem
from pyscheduling_cc.Problem import Solver
import pyscheduling_cc.JS.JobShop as JobShop


@dataclass
class JmCmax_Instance(JobShop.JobShopInstance):
    P: list[list[int]] = field(default_factory=list)  # Processing time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            JmCmax_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of configuration
        m = int(ligne0[1])  # number of jobs
        i = 1
        instance = cls("test", n, m)
        instance.P, i = instance.read_P(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int, configuration_number: int, protocol: JobShop.GenerationProtocol = JobShop.GenerationProtocol.VALLADA, law: JobShop.GenerationLaw = JobShop.GenerationLaw.UNIFORM, Pmin: int = -1, Pmax: int = -1, InstanceName: str = ""):
        """Random generation of RmSijkCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            configuration_number (int): number of machines of the instance
            protocol (JobShop.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to JobShop.GenerationProtocol.VALLADA.
            law (JobShop.GenerationLaw, optional): probablistic law of generation. Defaults to JobShop.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            JmCmax_Instance: the randomly generated instance
        """
        if(Pmin == -1):
            Pmin = randint(1, 100)
        if(Pmax == -1):
            Pmax = randint(Pmin, 100)
        instance = cls(InstanceName, jobs_number, configuration_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+" "+str(self.m)+"\n")
        for job in self.P:
            for operation in job:
                f.write(str(operation[0])+"\t"+str(operation[1])+"\t")
            f.write("\n")
        f.close()

    def init_sol_method(self):
        #return Heuristics.shifting_bottleneck
        pass

    def get_objective(self):
        return RootProblem.Objective.Cmax