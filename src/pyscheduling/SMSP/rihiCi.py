from dataclasses import dataclass, field
from functools import partial
from math import exp
from pathlib import Path
from random import randint, uniform
from time import perf_counter
from typing import Callable, List

import pyscheduling.Problem as RootProblem
import pyscheduling.SMSP.SingleMachine as SingleMachine
import pyscheduling.SMSP.SM_methods as Methods


@dataclass
class rihiCi_Instance(SingleMachine.SingleInstance):
    P: List[int] = field(default_factory=list)  # Processing time
    R: List[int] = field(default_factory=list) # release time
    Precedence : List[tuple] = field(default_factory=list) # precedence constraint
    external_params : int = 0
    W : List[int] = field(default_factory=list)
    D: List[List[int]] = field(default_factory=list) # due time

    def copy(self):
        return rihiCi_Instance(str(self.name),self.n,list(self.P),list(self.R),list(self.D))

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            riPrecLmax_Instance: created instance

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of jobs
        due_dates = int(ligne0[1])
        i = 1
        instance = cls("test", n)
        instance.external_params = due_dates
        instance.W, i = instance.read_1D(content, i)
        instance.P, i = instance.read_1D(content, i)
        instance.R, i = instance.read_1D(content, i)
        instance.D, i = instance.read_2D(n ,content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int,  protocol: SingleMachine.GenerationProtocol = SingleMachine.GenerationProtocol.BASE, law: SingleMachine.GenerationLaw = SingleMachine.GenerationLaw.UNIFORM, Pmin: int = 1, Pmax: int = -1, Wmin : int = 1, Wmax : int = 1, alpha : float = 0.0, due_time_factor : float = 0.0, InstanceName: str = ""):
        """Random generation of riPrecLmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            protocol (SingleMachine.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to SingleMachine.GenerationProtocol.VALLADA.
            law (SingleMachine.GenerationLaw, optional): probablistic law of generation. Defaults to SingleMachine.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to 1.
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
        instance.W = instance.generate_W(protocol, law, Wmin, Wmax)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.R = instance.generate_R(protocol,law,instance.P,Pmin,Pmax,alpha)
        for due in instance.D:
            due = instance.generate_D(protocol,law,instance.P,Pmin,Pmax,due_time_factor,RJobs=instance.R) 
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+" "+str(self.local_due_dates_number))
        f.write("\nWeights\n")
        for i in range(self.n):
            f.write(str(self.W[i])+"\t")
        f.write("\nProcessing time\n")
        for i in range(self.n):
            f.write(str(self.P[i])+"\t")
        f.write("\nRelease time\n")
        for i in range(self.n):
            f.write(str(self.R[i])+"\t")
        f.write("\nDue time\n")
        for i in range(self.n):
            for j in range(len(self.D[i])):
                f.write(str(self.D[i][j])+"\t")
            print("\n")
        f.close()        

    def get_objective(self):
        """to get the objective tackled by the instance

        Returns:
            RootProblem.Objective: Maximal Lateness
        """
        return RootProblem.Objective.wiTi

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.ACT


class Heuristics():

    def dynamic_dispatch_rule_with_precedence(instance : rihiCi_Instance, rule : Callable, reverse: bool = False):
        """Orders the jobs respecting the filter according to the rule. 
        The order is dynamic since it is determined each time a new job is inserted

        Args:
            instance (SingleInstance): Instance to be solved
            rule (Callable): a lambda function that defines the sorting criteria taking the instance and job_id as the parameters
            filter (Callable): a lambda function that defines a filter condition taking the instance, job_id and current time as the parameters
            reverse (bool, optional): flag to sort in decreasing order. Defaults to False.

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        startTime = perf_counter()
        solution = SingleMachine.SingleSolution(instance)

        remaining_jobs_list = list(range(instance.n))
        ci = 0
        
        insert_idx = 0
        while(len(remaining_jobs_list)>0):
            remaining_jobs_list.sort(key= partial(rule, instance, ci), reverse=reverse)
            for job in remaining_jobs_list :
                legal = True
                for precedence_couple in instance.Precedence :
                    if precedence_couple[1]==job and precedence_couple[0] in remaining_jobs_list :
                        legal = False
                        break
                if legal : 
                    taken_job = job
                    break
            #ci = solution.machine.objective_insert(taken_job, insert_idx, instance)
            start_time = max(instance.R[taken_job],ci)
            ci = start_time + instance.P[taken_job]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,start_time,ci))
            solution.machine.objective_value = ci
            remaining_jobs_list.remove(taken_job)
            insert_idx += 1

        solution.objective_value = ci
        
        return RootProblem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])


    def ACT(instance : rihiCi_Instance):
        avgP = int(sum(instance.P)/len(instance.P))
        K = 0.1
        sorting_rule = lambda instance,t,job_id : sum([(float(instance.W[k])/float(instance.P[job_id]))*exp(-max(instance.D[job_id][k]-instance.P[job_id]+(instance.R[job_id]-t),0)/(K*avgP)) for k in range(instance.external_params)])
        return Heuristics.dynamic_dispatch_rule_with_precedence(instance,sorting_rule,True)


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
