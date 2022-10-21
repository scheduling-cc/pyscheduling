import sys
from dataclasses import dataclass, field
from random import randint, uniform
from pathlib import Path
from time import perf_counter

import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Solver
import pyscheduling.SMSP.SingleMachine as SingleMachine
import pyscheduling.SMSP.SM_Methods as Methods


@dataclass
class riPrecLmax_Instance(SingleMachine.SingleInstance):
    P: list[int] = field(default_factory=list)  # Processing time
    R: list[int] = field(default_factory=list) # release time
    D: list[int] = field(default_factory=list) # due time
    Precedence : list[tuple] = field(default_factory=list) # precedence constraint

    def copy(self):
        return riPrecLmax_Instance(str(self.name),self.n,list(self.P),list(self.R),list(self.D))

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
        i = 1
        instance = cls("test", n)
        instance.P, i = instance.read_P(content, i)
        instance.R, i = instance.read_R(content, i)
        instance.D, i = instance.read_D(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int,  protocol: SingleMachine.GenerationProtocol = SingleMachine.GenerationProtocol.BASE, law: SingleMachine.GenerationLaw = SingleMachine.GenerationLaw.UNIFORM, Pmin: int = 1, Pmax: int = -1, alpha : float = 0.0, due_time_factor : float = 0.0, InstanceName: str = ""):
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

    def LB_preemptive_EDD(self, start_time : int = 0, jobs_list : list[int] = None):
        """returns the objective value returned by applying the preemptive EDD rule on the instance
        object from a given starting time and remaining jobs list to be scheduled

        Args:
            start_time (int, optional): Instant of the beginning of the schedule. Defaults to 0.
            jobs_list (list[int], optional): Remaining jobs list to be scheduled. Defaults to None.

        Returns:
            int: lower bound of the instance
        """
        if jobs_list is None: remaining_job_list = list(range(self.n))
        remaining_job_list = list(jobs_list)
        release_time = [self.R[job] for job in jobs_list]
        release_time_dict = dict(zip(remaining_job_list, release_time))
        processing_time = list(self.P)
        release_time.sort()
        maximum_lateness = 0
        start_index = 0
        
        while(release_time[start_index]<start_time) : 
            start_index += 1
            if start_index == len(release_time) :
                start_index = len(release_time)
                break
        t = max(start_time,release_time[0])
        for instant in range(start_index-1,len(release_time)-1):
            remaining_job_list_released = [job for job in remaining_job_list if release_time_dict[job]<=t]
            remaining_job_list_released.sort(key = lambda job_id : self.D[job_id])
            
            while(t < release_time[instant+1]):
                if len(remaining_job_list_released) == 0 : break
                job_id = remaining_job_list_released.pop(0)
                exec_time = min(t+processing_time[job_id],release_time[instant+1]) - t
                t += exec_time
                processing_time[job_id] = processing_time[job_id] - exec_time
                if processing_time[job_id] == 0 : 
                    remaining_job_list.remove(job_id)
                    maximum_lateness = max(maximum_lateness,max(t-self.D[job_id],0))

        remaining_job_list_released = [job for job in remaining_job_list if release_time_dict[job]<=t]
        remaining_job_list_released.sort(key = lambda job_id : self.D[job_id])
        
        while len(remaining_job_list) > 0:
            job_id = remaining_job_list_released.pop(0)
            t += processing_time[job_id]
            processing_time[job_id] = 0
            if processing_time[job_id] == 0 : 
                remaining_job_list.remove(job_id)
                maximum_lateness = max(maximum_lateness,max(t-self.D[job_id],0))

        return maximum_lateness

    def get_objective(self):
        """to get the objective tackled by the instance

        Returns:
            RootProblem.Objective: Maximal Lateness
        """
        return RootProblem.Objective.Lmax

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return None


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
        """Branching rule from Pinedo book page 44

        Args:
            node (RootProblem.Branch_Bound.Node): node to branch from
        """
        if node.partial_solution is None : 
            remaining_job_list = [job for job in list(range(self.instance.n))]
            for precedence in self.instance.Precedence :
                remaining_job_list.remove(precedence[1])
            partial_solution_len = 0
            t = 0
            node.partial_solution = []
        else : 
            partial_solution_job_id = [job.id for job in node.partial_solution]
            remaining_job_list_tmp = [job for job in list(range(self.instance.n)) if job not in partial_solution_job_id]
            jobs_blocked_by_precedence = []
            for precedence in self.instance.Precedence :
                if precedence[0] not in partial_solution_job_id : jobs_blocked_by_precedence.append(precedence[1])
            remaining_job_list = [job for job in remaining_job_list_tmp if job not in jobs_blocked_by_precedence]
            partial_solution_len = len(node.partial_solution)
            t = node.partial_solution[partial_solution_len-1].end_time
        factor = None
        for job in remaining_job_list:
            calculated_factor = max(t,self.instance.R[job])+self.instance.P[job]
            if factor is None or calculated_factor < factor:
                factor = calculated_factor
        node.sub_nodes = []
        if partial_solution_len == self.instance.n - 1 : if_solution = True
        else: if_solution = False
        for job in remaining_job_list:
            if self.instance.R[job] < factor :
                startTime = max(t,self.instance.R[job])
                new_partial_solution = node.partial_solution+[SingleMachine.Job(job,startTime,startTime+self.instance.P[job])]
                sub_node = self.Node(if_solution=if_solution,partial_solution=new_partial_solution)
                node.sub_nodes.append(sub_node)

    def bound(self, node : RootProblem.Branch_Bound.Node):
        """affects the preemptive_EDD value to the lower bound attribute of the node

        Args:
            node (RootProblem.Branch_Bound.Node): the node to bound
        """
        maximum_lateness = self.objective(node)
        partial_solution_job_id = [job.id for job in node.partial_solution]
        remaining_jobs_list = [job for job in list(range(self.instance.n)) if job not in partial_solution_job_id]
        startTime = node.partial_solution[len(node.partial_solution)-1].end_time
        maximum_lateness = max(maximum_lateness,self.instance.LB_preemptive_EDD(startTime,remaining_jobs_list))
        node.lower_bound = maximum_lateness

    def objective(self, node : RootProblem.Branch_Bound.Node):
        """Objective value evaluator

        Args:
            node (RootProblem.Branch_Bound.Node): node to be evaluated as a solution

        Returns:
            int: maximum lateness
        """
        maximum_lateness = 0
        for job in node.partial_solution:
            maximum_lateness = max(maximum_lateness,max(job.end_time-self.instance.D[job.id],0))
        return maximum_lateness

    def solution_format(self, partial_solution: object, objective_value):
        solution = SingleMachine.SingleSolution(self.instance)
        solution.machine.job_schedule = partial_solution
        solution.machine.objective = objective_value
        solution.machine.last_job = partial_solution[len(partial_solution)-1].id
        solution.objective_value = objective_value
        return solution