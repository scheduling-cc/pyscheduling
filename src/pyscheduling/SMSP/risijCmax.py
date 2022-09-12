from math import exp
import sys
from dataclasses import dataclass, field
from random import randint, uniform
from pathlib import Path
from time import perf_counter

import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Solver
import pyscheduling.SMSP.SingleMachine as SingleMachine
import pyscheduling.SMSP.SM_Methods as Methods
from pyscheduling.SMSP.SM_Methods import ExactSolvers


@dataclass
class risijCmax_Instance(SingleMachine.SingleInstance):
    P: list[int] = field(default_factory=list)  # Processing time
    R: list[int] = field(default_factory=list)
    S: list[list[int]] = field(default_factory=list) # Setup time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            risijCmax_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of jobs
        i = 1
        instance = cls("test", n)
        instance.P, i = instance.read_P(content, i)
        instance.R, i = instance.read_P(content, i)
        instance.S, i = instance.read_S(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int,  protocol: SingleMachine.GenerationProtocol = SingleMachine.GenerationProtocol.BASE, law: SingleMachine.GenerationLaw = SingleMachine.GenerationLaw.UNIFORM, Pmin: int = 1, Pmax: int = -1, alpha : float = 0.0, Gamma : float = 0.0, Smin : int = -1, Smax : int = -1, InstanceName: str = ""):
        """Random generation of risijCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            protocol (SingleMachine.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to SingleMachine.GenerationProtocol.VALLADA.
            law (SingleMachine.GenerationLaw, optional): probablistic law of generation. Defaults to SingleMachine.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to 1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            alpha (float, optional): Release time factor. Defaults to 0.0.
            Gamma (float, optional): Setup time factor. Defaults to 0.0.
            Smin (int, optional) : Minimal setup time. Defaults to -1.
            Smax (int, optional) : Maximal setup time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            risijCmax_Instance: the randomly generated instance
        """
        if(Pmax == -1):
            Pmax = Pmin + randint(1, 100)
        if(alpha == 0.0):
            alpha = round(uniform(1.0, 3.0), 1)
        if(Gamma == 0.0):
            Gamma = round(uniform(1.0, 3.0), 1)
        if(Smin == -1):
            Smin = randint(1, 100)
        if(Smax == -1):
            Smax = randint(Smin, 100)
        instance = cls(InstanceName, jobs_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.R = instance.generate_R(protocol,law,instance.P,Pmin,Pmax,alpha)
        instance.S = instance.generate_S(protocol,law,instance.P,Gamma,Smin,Smax)
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
        f.write("\nSSD\n")
        for i in range(self.n):
            for j in range(self.n):
                f.write(str(self.S[i][j])+"\t")
            f.write("\n")
        f.close()

    def get_objective(self):
        """to get the objective tackled by the instance

        Returns:
            RootProblem.Objective: Makespan
        """
        return RootProblem.Objective.Cmax

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.constructive


class Heuristics(Methods.Heuristics_Cmax):
    
    
    @staticmethod
    def constructive(instance: risijCmax_Instance):
        """the greedy constructive heuristic to find an initial solution of risijCmax problem minimalizing the factor of (processing time + setup time) of the job to schedule at a given time

        Args:
            instance (risijCmax_Instance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = SingleMachine.SingleSolution(instance=instance)
        remaining_jobs_list = [i for i in range(instance.n)]
        while len(remaining_jobs_list) != 0:
            min_factor = None
            for i in remaining_jobs_list:
                current_machine_schedule = solution.machine
                if (current_machine_schedule.last_job == -1):
                    startTime = max(current_machine_schedule.objective,
                                    instance.R[i])
                    factor = startTime + instance.P[i] + \
                        instance.S[i][i]  # Added Sj_ii for rabadi
                else:
                    startTime = max(current_machine_schedule.objective,
                                    instance.R[i])
                    factor = startTime + instance.P[i] + instance.S[
                        current_machine_schedule.last_job][i]

                if not min_factor or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
                    taken_startTime = startTime
            if (solution.machine.last_job == -1):
                ci = taken_startTime + instance.P[taken_job] + \
                    instance.S[taken_job][taken_job]  # Added Sj_ii for rabadi
            else:
                ci = taken_startTime + instance.P[taken_job]+ instance.S[
                        solution.machine.last_job][taken_job]
            solution.machine.objective = ci
            solution.machine.last_job = taken_job
            solution.machine.job_schedule.append(
                SingleMachine.Job(taken_job, taken_startTime, min_factor))
            remaining_jobs_list.remove(taken_job)
            if (ci > solution.objective_value):
                solution.objective_value = ci

        return RootProblem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])


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

