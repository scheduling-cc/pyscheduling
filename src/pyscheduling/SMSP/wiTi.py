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
class wiTi_Instance(SingleMachine.SingleInstance):
    W : list[int] = field(default_factory=list) # Jobs weights
    P: list[int] = field(default_factory=list)  # Processing time
    D: list[int] = field(default_factory=list) # Due time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            wiTi_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of jobs
        i = 1
        instance = cls("test", n)
        instance.P, i = instance.read_P(content, i)
        instance.W, i = instance.read_W(content, i)
        instance.D, i = instance.read_D(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int,  protocol: SingleMachine.GenerationProtocol = SingleMachine.GenerationProtocol.BASE, law: SingleMachine.GenerationLaw = SingleMachine.GenerationLaw.UNIFORM, Wmin : int = 1, Wmax : int = 1 ,Pmin: int = 1, Pmax: int = -1, due_time_factor : float = 0.0, InstanceName: str = ""):
        """Random generation of wiTi problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            protocol (SingleMachine.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to SingleMachine.GenerationProtocol.VALLADA.
            law (SingleMachine.GenerationLaw, optional): probablistic law of generation. Defaults to SingleMachine.GenerationLaw.UNIFORM.
            Wmin (int, optional): Minimal weight. Defaults to 1.
            Wmax (int, optional): Maximal weight. Defaults to 1.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            due_time_factor (float, optional): Due time factor. Defaults to 0.0.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            wiTi_Instance: the randomly generated instance
        """
        if(Pmax == -1):
            Pmax = Pmin + randint(1, 100)
        if(due_time_factor == 0.0):
            due_time_factor = round(uniform(0, 1), 1)
        instance = cls(InstanceName, jobs_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.W = instance.generate_W(protocol,law, Wmin, Wmax)
        instance.D = instance.generate_D(protocol,law,instance.P,Pmin,Pmax,due_time_factor)
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
        f.write("\nWeights\n")
        for i in range(self.n):
            f.write(str(self.W[i])+"\t")
        f.write("\nDue time\n")
        for i in range(self.n):
            f.write(str(self.D[i])+"\t")
        f.close()

    def get_objective(self):
        """to get the objective tackled by the instance

        Returns:
            RootProblem.Objective: Total wighted Lateness
        """
        return RootProblem.Objective.wiTi

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.ACT


class Heuristics():

    @staticmethod
    def WSPT(instance : wiTi_Instance):
        """WSPT rule is efficient if the due dates are too tight (for overdue jobs)

        Args:
            instance (wiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        startTime = perf_counter()
        jobs = list(range(instance.n))
        jobs.sort(reverse=True,key=lambda job_id : float(instance.W[job_id])/float(instance.P[job_id]))
        solution = SingleMachine.SingleSolution(instance)
        for job in jobs:
            solution.machine.job_schedule.append(SingleMachine.Job(job,0,0)) 
        solution.machine.total_weighted_lateness(instance)
        solution.fix_objective()
        return RootProblem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])

    @staticmethod
    def MS(instance : wiTi_Instance):
        """MS rule is efficient if the due dates are too loose (for not overdue jobs)

        Args:
            instance (wiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        startTime = perf_counter()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiTi_index = []
        ci = 0
        wiTi = 0
        remaining_jobs_list = list(range(instance.n))
        rule = lambda job_id : max(instance.D[job_id]-instance.P[job_id]-ci,0)
        while(len(remaining_jobs_list)>0):
            remaining_jobs_list.sort(reverse=False,key=rule)
            taken_job = remaining_jobs_list[0]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,ci,ci+instance.P[taken_job]))
            ci += instance.P[taken_job]
            wiTi += instance.W[taken_job]*max(ci-instance.D[taken_job],0)
            solution.machine.wiTi_index.append(wiTi)
            remaining_jobs_list.pop(0)
        solution.machine.objective=solution.machine.wiTi_index[instance.n-1]
        solution.fix_objective()
        return RootProblem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])
    
    @staticmethod
    def ACT(instance : wiTi_Instance):
        """Appearant Cost Tardiness rule balances between WSPT and MS rules based on due dates tightness and range

        Args:
            instance (wiTi_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        startTime = perf_counter()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiTi_index = []
        ci = 0
        wiTi = 0
        remaining_jobs_list = list(range(instance.n))
        sumP = sum(instance.P)
        K = Heuristics_Tuning.ACT(instance)
        rule = lambda job_id : (float(instance.W[job_id])/float(instance.P[job_id]))*exp(-max(instance.D[job_id]-instance.P[job_id]-ci,0)/(K*sumP))
        while(len(remaining_jobs_list)>0):
            remaining_jobs_list.sort(reverse=True,key=rule)
            taken_job = remaining_jobs_list[0]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,ci,ci+instance.P[taken_job]))
            ci += instance.P[taken_job]
            wiTi += instance.W[taken_job]*max(ci-instance.D[taken_job],0)
            solution.machine.wiTi_index.append(wiTi)
            remaining_jobs_list.pop(0)
        solution.machine.objective=solution.machine.wiTi_index[instance.n-1]
        solution.fix_objective()
        return RootProblem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])

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

class Heuristics_Tuning():

    @staticmethod
    def ACT(instance : wiTi_Instance):
        """Analyze the instance to consequently tune the ACT. For now, the tuning is static.

        Args:
            instance (riwiTi_Instance): Instance tackled by ACT heuristic

        Returns:
            int, int: K
        """
        Tightness = 1 - sum(instance.D)/(instance.n*sum(instance.P))
        Range = (max(instance.D)-min(instance.D))/sum(instance.P)
        return 0.2