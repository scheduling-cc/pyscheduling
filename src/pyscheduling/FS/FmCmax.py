from dataclasses import dataclass, field
from pathlib import Path
from random import randint

from time import perf_counter


import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Solver
import pyscheduling.FS.FlowShop as FlowShop
import pyscheduling.FS.FS_Methods as FS_Methods


@dataclass
class FmCmax_Instance(FlowShop.FlowShopInstance):
    P: list[list[int]] = field(default_factory=list)  # Processing time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            FmCmax_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of configuration
        m = int(ligne0[2])  # number of jobs
        i = 2
        instance = cls("test", n, m)
        instance.P, i = instance.read_P(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int, configuration_number: int, protocol: FlowShop.GenerationProtocol = FlowShop.GenerationProtocol.VALLADA, law: FlowShop.GenerationLaw = FlowShop.GenerationLaw.UNIFORM, Pmin: int = -1, Pmax: int = -1, InstanceName: str = ""):
        """Random generation of FmCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            configuration_number (int): number of machines of the instance
            protocol (FlowShop.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to FlowShop.GenerationProtocol.VALLADA.
            law (FlowShop.GenerationLaw, optional): probablistic law of generation. Defaults to FlowShop.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            FmCmax_Instance: the randomly generated instance
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
        f.write(str(self.n)+"  "+str(self.m)+"\n")
        f.write(str(self.m)+"\n")
        for i in range(self.n):
            for j in range(self.m):
                f.write("\t"+str(j)+"\t"+str(self.P[i][j]))
            f.write("\n")
        f.close()

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.slope

    def get_objective(self):
        """to get the objective tackled by the instance

        Returns:
            RootProblem.Objective: Makespan
        """
        return RootProblem.Objective.Cmax


class Heuristics(FS_Methods.Heuristics_Cmax):

    @staticmethod
    def slope(instance: FmCmax_Instance):
        """Inspired from Jonhson's rule, this heuristic schedules first the jobs with the smallest processing times on the first machines

        Args:
            instance (FmCmax_Instance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = FlowShop.FlowShopSolution(instance=instance)
        jobs = list(range(instance.n))
        # m+1 to translate the set of numbers of m from [[0,m-1]] to [[1,m]]
        # machine_id+1 to translate the set of numbers of machine_id from [[0,m-1]] to [[1,m]]
        slope_index = lambda job_id : -sum([((instance.m + 1) - (2*(machine_id+1)-1))*instance.P[job_id][machine_id] for machine_id in range(instance.m)])
        jobs.sort(reverse=True,key=slope_index)
        solution.job_schedule = jobs
        solution.cmax()
        return RootProblem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]