import sys
from dataclasses import dataclass, field
from random import randint, uniform
from pathlib import Path
from time import perf_counter

from matplotlib import pyplot as plt

import pyscheduling_cc.Problem as Problem
from pyscheduling_cc.Problem import Solver
import pyscheduling_cc.SMSP.SingleMachine as SingleMachine
from pyscheduling_cc.SMSP.SingleMachine import ExactSolvers


@dataclass
class wiTi_Instance(SingleMachine.SingleInstance):
    W : list[int] = field(default_factory=list) # Jobs weights
    P: list[int] = field(default_factory=list)  # Processing time
    W: list[int] = field(default_factory=list) # Due time

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
        """Random generation of RmSijkCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            protocol (SingleMachine.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to SingleMachine.GenerationProtocol.VALLADA.
            law (SingleMachine.GenerationLaw, optional): probablistic law of generation. Defaults to SingleMachine.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
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

    def create_solution(self):
        return wiTi_Solution(self)


@dataclass
class wiTi_Solution(SingleMachine.SingleSolution):

    def __init__(self, instance: wiTi_Instance = None, machine : SingleMachine.Machine = None, objective_value: int = 0):
        """Constructor of wiTi_Solution

        Args:
            instance (wiTi_Instance, optional): Instance to be solved by the solution. Defaults to None.
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
        copy_solution = wiTi_Solution(self.instance)
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
                prevEndTime = 0
                for element in schedule:
                    job_index, startTime, endTime = element
                    if prevEndTime < startTime:
                        # Idle Time
                        gnt.broken_barh(
                            [(prevEndTime, startTime - prevEndTime)], (10, 9), facecolors=('tab:gray'))
                    
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

class Heuristics():
    
    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]


class Metaheuristics():
    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

