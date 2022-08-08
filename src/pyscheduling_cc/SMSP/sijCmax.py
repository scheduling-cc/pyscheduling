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
from pyscheduling_cc.SMSP.SingleMachine import ExactSolvers


@dataclass
class sijCmax_Instance(SingleMachine.SingleInstance):
    P: list[int] = field(default_factory=list)  # Processing time
    S: list[list[int]] = field(default_factory=list) # Setup time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            sijCmax_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of jobs
        i = 1
        instance = cls("test", n)
        instance.P, i = instance.read_P(content, i)
        instance.S, i = instance.read_S(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int,  protocol: SingleMachine.GenerationProtocol = SingleMachine.GenerationProtocol.BASE, law: SingleMachine.GenerationLaw = SingleMachine.GenerationLaw.UNIFORM, Pmin: int = 1, Pmax: int = -1, Gamma : float = 0.0, Smin : int = -1, Smax : int = -1, InstanceName: str = ""):
        """Random generation of RmSijkCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            protocol (SingleMachine.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to SingleMachine.GenerationProtocol.VALLADA.
            law (SingleMachine.GenerationLaw, optional): probablistic law of generation. Defaults to SingleMachine.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            sijCmax_Instance: the randomly generated instance
        """
        if(Pmax == -1):
            Pmax = Pmin + randint(1, 100)
        if(Gamma == 0.0):
            Gamma = round(uniform(1.0, 3.0), 1)
        if(Smin == -1):
            Smin = randint(1, 100)
        if(Smax == -1):
            Smax = randint(Smin, 100)
        instance = cls(InstanceName, jobs_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
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
        f.write("\nSSD\n")
        for i in range(self.n):
            for j in range(self.n):
                f.write(str(self.S[i][j])+"\t")
            f.write("\n")
        f.close()

    def get_objective(self):
        return "Cmax"
    
    def create_solution(self):
        return sijCmax_Solution(self)

    def init_sol_method(self):
        return Heuristics.constructive


@dataclass
class sijCmax_Solution(SingleMachine.SingleSolution):

    def __init__(self, instance: sijCmax_Instance = None, machine : SingleMachine.Machine = None, objective_value: int = 0):
        """Constructor of sijCmax_Solution

        Args:
            instance (sijCmax_Instance, optional): Instance to be solved by the solution. Defaults to None.
            configuration (SingleMachine.Machine, optional): list of machines of the instance. Defaults to None.
            objective_value (int, optional): initial objective value of the solution. Defaults to 0.
        """
        self.instance = instance
        if machine is None:
            self.machine = SingleMachine.Machine(0, -1, [])
        else:
            self.machine = machine
        self.objective_value = objective_value

    def __str__(self):
        return "Cmax : " + str(self.objective_value) + "\n" + "Job_schedule (job_id , start_time , completion_time) | Completion_time\n" + self.machine.__str__()

    def copy(self):
        copy_solution = sijCmax_Solution(self.instance)
        copy_solution.machine = self.machine.copy()
        copy_solution.objective_value = self.objective_value
        return copy_solution

    @classmethod
    def read_txt(cls, path: Path):
        """Read a solution from a txt file

        Args:
            path (Path): path to the solution's txt file of type Path from pathlib

        Returns:
            RmSijkCmax_Solution:
        """
        f = open(path, "r")
        content = f.read().split('\n')
        objective_value_ = int(content[0].split(':')[1])
        line_content = content[2].split('|')
        machine = SingleMachine.Machine(int(line_content[1]), job_schedule=[SingleMachine.Job(
                int(j[0]), int(j[1]), int(j[2])) for j in [job.strip()[1:len(job.strip())-1].split(',') for job in line_content[0].split(':')]])
        solution = cls(objective_value=objective_value_,
                       machine=machine)
        return solution

    def plot(self, path: Path = None) -> None:
        """Plot the solution in an appropriate diagram"""
        if "matplotlib" in sys.modules:
            if self.instance is not None:
                # Add Tasks ID
                fig, gnt = plt.subplots()

                # Setting labels for x-axis and y-axis
                gnt.set_xlabel('seconds')
                gnt.set_ylabel('Machines')

                # Setting ticks on y-axis

                ticks = []
                ticks_labels = []
                ticks.append(15)

                gnt.set_yticks(ticks)
                # Labelling tickes of y-axis
                gnt.set_yticklabels(ticks_labels)

                # Setting graph attribute
                gnt.grid(True)

                schedule = self.machine.job_schedule

                prev = -1
                prevEndTime = 0
                for element in schedule:
                    job_index, startTime, endTime = element
                    if prevEndTime < startTime:
                        # Idle Time
                        gnt.broken_barh(
                            [(prevEndTime, startTime - prevEndTime)], (10, 9), facecolors=('tab:gray'))
                    if prev != -1:
                            # Setup Time
                        gnt.broken_barh([(startTime, self.instance.S[prev][job_index])], (
                            10, 9), facecolors=('tab:orange'))
                        # Processing Time
                        gnt.broken_barh([(startTime + self.instance.S[prev][job_index],
                                        self.instance.P[job_index])], (10, 9), facecolors=('tab:blue'))
                    else:
                        gnt.broken_barh([(startTime, self.instance.P[job_index])], (
                            10, 9), facecolors=('tab:blue'))
                    prev = job_index
                    prevEndTime = endTime
                if path:
                    plt.savefig(path)
                else:
                    plt.show()
                return
            else:
                print("Please assign the solved instance to the solution object")
        else:
            print("Matplotlib is not installed, you can't use gant_plot")
            return

    def is_valid(self):
        """
        Check if solution respects the constraints
        """
        set_jobs = set()
        is_valid = True
        ci, expected_start_time = 0, 0
        for i, element in enumerate(self.machine.job_schedule):
            job, startTime, endTime = element
            # Test End Time + start Time
            expected_start_time = ci
            ci +=  self.instance.P[job]

            if startTime != expected_start_time or endTime != ci:
                print(f'## Error:  found {element} expected {job,expected_start_time, ci}')
                is_valid = False
            set_jobs.add(job)

        is_valid &= len(set_jobs) == self.instance.n
        return is_valid

class Heuristics(Methods.Heuristics_Cmax):
    
    
    @staticmethod
    def constructive(instance: sijCmax_Instance):
        """the greedy constructive heuristic to find an initial solution of RmSijkCmax problem minimalizing the factor of (processing time + setup time) of the job to schedule at a given time

        Args:
            instance (RmSijkCmax_Instance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = sijCmax_Solution(instance=instance)
        remaining_jobs_list = [i for i in range(instance.n)]
        while len(remaining_jobs_list) != 0:
            min_factor = None
            for i in remaining_jobs_list:

                current_machine_schedule = solution.machine
                if (current_machine_schedule.last_job == -1):
                    factor = current_machine_schedule.objective + \
                        instance.P[i]
                else:
                    factor = current_machine_schedule.objective + \
                        instance.P[i] + \
                        instance.S[current_machine_schedule.last_job][i]

                if not min_factor or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
            if (solution.machine.last_job == -1):
                ci = solution.machine.objective + \
                    instance.P[taken_job]
            else:
                ci = solution.machine.objective + instance.P[taken_job]+ \
                    instance.S[solution.machine.last_job][taken_job]

            solution.machine.job_schedule.append(SingleMachine.Job(
                taken_job, solution.machine.objective, ci))
            solution.machine.objective = ci
            solution.machine.last_job = taken_job

            remaining_jobs_list.remove(taken_job)
            if (ci > solution.objective_value):
                solution.objective_value = ci

        return Problem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])


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

