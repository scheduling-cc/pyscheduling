from math import exp
import sys
from dataclasses import dataclass, field
from random import randint, uniform
from pathlib import Path
from time import perf_counter

from matplotlib import pyplot as plt

import pyscheduling_cc.Problem as Problem
from pyscheduling_cc.Problem import Solver
import pyscheduling_cc.SMSP.SingleMachine as SingleMachine
import pyscheduling_cc.SMSP.SM_Methods as Methods
from pyscheduling_cc.SMSP.SM_Methods import ExactSolvers


@dataclass
class risijwiTi_Instance(SingleMachine.SingleInstance):
    W : list[int] = field(default_factory=list) # Jobs weights
    P: list[int] = field(default_factory=list)  # Processing time
    R: list[int] = field(default_factory=list) # release time
    D: list[int] = field(default_factory=list) # Due time
    S: list[list[int]] = field(default_factory=list) # Setup time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            risijwiTi_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of jobs
        i = 1
        instance = cls("test", n)
        instance.P, i = instance.read_P(content, i)
        instance.W, i = instance.read_W(content, i)
        instance.R, i = instance.read_R(content, i)
        instance.D, i = instance.read_D(content, i)
        instance.S, i = instance.read_S(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int,  protocol: SingleMachine.GenerationProtocol = SingleMachine.GenerationProtocol.BASE, law: SingleMachine.GenerationLaw = SingleMachine.GenerationLaw.UNIFORM, Wmin : int = 1, Wmax : int = 1 ,Pmin: int = 1, Pmax: int = -1, alpha : float = 0.0, due_time_factor : float = 0.0, Gamma : float = 0.0, Smin : int = -1, Smax : int = -1, InstanceName: str = ""):
        """Random generation of RmSijkCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            protocol (SingleMachine.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to SingleMachine.GenerationProtocol.VALLADA.
            law (SingleMachine.GenerationLaw, optional): probablistic law of generation. Defaults to SingleMachine.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            risijwiTi_Instance: the randomly generated instance
        """
        if(Pmax == -1):
            Pmax = Pmin + randint(1, 100)
        if(alpha == 0.0):
            alpha = round(uniform(5.0, 10.0), 1)
        if(due_time_factor == 0.0):
            due_time_factor = round(uniform(0, 1), 1)
        if(Gamma == 0.0):
            Gamma = round(uniform(1.0, 3.0), 1)
        if(Smin == -1):
            Smin = randint(1, 100)
        if(Smax == -1):
            Smax = randint(Smin, 100)
        instance = cls(InstanceName, jobs_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.W = instance.generate_W(protocol,law, Wmin, Wmax)
        instance.R = instance.generate_R(protocol,law,instance.P,Pmin,Pmax,alpha)
        instance.D = instance.generate_D(protocol,law,instance.P,Pmin,Pmax,due_time_factor,RJobs=instance.R)
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
        f.write("\nWeights\n")
        for i in range(self.n):
            f.write(str(self.W[i])+"\t")
        f.write("\nRelease time\n")
        for i in range(self.n):
            f.write(str(self.R[i])+"\t")
        f.write("\nDue time\n")
        for i in range(self.n):
            f.write(str(self.D[i])+"\t")
        f.write("\nSSD\n")
        for i in range(self.n):
            for j in range(self.n):
                f.write(str(self.S[i][j])+"\t")
            f.write("\n")
        f.close()

    def get_objective(self):
        return SingleMachine.Objective.wiTi

    def init_sol_method(self):
        return Heuristics.ACTS_WSECi


class Heuristics():
    
    @staticmethod
    def ACTS_WSECi(instance : risijwiTi_Instance):
        startTime = perf_counter()
        solution = SingleMachine.SingleSolution(instance)
        solution.machine.wiTi_index = []
        ci = 0
        wiTi = 0
        prev_job = -1
        remaining_jobs_list = list(range(instance.n))
        while(len(remaining_jobs_list)>0):
            prev_job, taken_job = Heuristics_HelperFunctions.ACTS_WSECi_Sorting(instance,remaining_jobs_list,ci,prev_job)
            start_time = max(instance.R[taken_job],ci)
            ci = start_time + instance.S[prev_job][taken_job] + instance.P[taken_job]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,start_time,ci))
            wiTi += instance.W[taken_job]*max(ci-instance.D[taken_job],0)
            solution.machine.wiTi_index.append(wiTi)
            remaining_jobs_list.remove(taken_job)
            prev_job = taken_job
        solution.machine.objective=solution.machine.wiTi_index[instance.n-1]
        solution.fix_objective()
        return Problem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])

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

class Heuristics_HelperFunctions():

    @staticmethod
    def ACTS_WSECi_Sorting(instance : risijwiTi_Instance, remaining_jobs : list[SingleMachine.Job], t : int, prev_job : int):
        sumP = sum(instance.P)
        sumS = 0
        for i in range(instance.n):
            sumSi = sum(instance.S[i])
            sumS += sumSi
        K1, K2 = Heuristics_HelperFunctions.ACTS_WSECi_Tuning(instance)
        rule = lambda prev_j,job_id : (float(instance.W[job_id])/float(max(instance.R[job_id] - t,0) + instance.P[job_id]))*exp(
            -max(instance.D[job_id]-instance.P[job_id]-t,0)/(K1*sumP))*exp(-instance.S[prev_j][job_id]/(K2*sumS))
        max_rule_value = -1
        if prev_job == -1:
            for job in remaining_jobs:
                rule_value = rule(job,job)
                if max_rule_value<rule_value: 
                    max_rule_value = rule_value
                    taken_job = job
            return taken_job, taken_job
        else:
            for job in remaining_jobs:
                rule_value = rule(prev_job,job)
                if max_rule_value<rule_value: 
                    max_rule_value = rule_value
                    taken_job = job
            return prev_job, taken_job
        

    @staticmethod
    def ACTS_WSECi_Tuning(instance : risijwiTi_Instance):
        Tightness = 1 - sum(instance.D)/(instance.n*sum(instance.P))
        Range = (max(instance.D)-min(instance.D))/sum(instance.P)
        return 0.2, 1