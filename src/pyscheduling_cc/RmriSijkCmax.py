import sys
from dataclasses import dataclass, field
from pathlib import Path
from random import randint, uniform

import matplotlib.pyplot as plt

import pyscheduling_cc.ParallelMachines as ParallelMachines
import pyscheduling_cc.Problem as Problem

@dataclass
class RmriSijkCmax_Instance(ParallelMachines.ParallelInstance):
    P: list[list[int]] = field(default_factory=list)  # Processing time
    S: list[list[list[int]]] = field(default_factory=list)  # Setup time
    R: list[int] = field(default_factory=list) # Release time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            RmSijkCmax_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of configuration
        m = int(ligne0[1])  # number of jobs
        i = 2
        instance = cls("test", n, m)
        instance.P, i = instance.read_P(content, i)
        instance.R, i = instance.read_R(content, i)
        instance.S, i = instance.read_S(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int, configuration_number: int, protocol: ParallelMachines.GenerationProtocol = ParallelMachines.GenerationProtocol.VALLADA, law: ParallelMachines.GenerationLaw = ParallelMachines.GenerationLaw.UNIFORM, Pmin: int = -1, Pmax: int = -1, Alpha : float = 0.0, Gamma: float = 0.0, Smin:  int = -1, Smax: int = -1, InstanceName: str = ""):
        """Random generation of RmSijkCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            configuration_number (int): number of machines of the instance
            protocol (ParallelMachines.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to ParallelMachines.GenerationProtocol.VALLADA.
            law (ParallelMachines.GenerationLaw, optional): probablistic law of generation. Defaults to ParallelMachines.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            Alpha (float,optional): Release time factor. Defaults to 0.0.
            Gamma (float, optional): Setup time factor. Defaults to 0.0.
            Smin (int, optional): Minimal setup time. Defaults to -1.
            Smax (int, optional): Maximal setup time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            RmSijkCmax_Instance: the randomly generated instance
        """
        if(Pmin == -1):
            Pmin = randint(1, 100)
        if(Pmax == -1):
            Pmax = randint(Pmin, 100)
        if(Alpha == 0.0):
            Alpha = round(uniform(1.0, 3.0), 1)
        if(Gamma == 0.0):
            Gamma = round(uniform(1.0, 3.0), 1)
        if(Smin == -1):
            Smin = randint(1, 100)
        if(Smax == -1):
            Smax = randint(Smin, 100)
        instance = cls(InstanceName, jobs_number, configuration_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.R = instance.generate_R(protocol, law, instance.P, Pmin, Pmax, Alpha)
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
        f.write(str(self.m)+"\n")
        for i in range(self.n):
            for j in range(self.m):
                f.write(str(self.P[i][j])+"\t")
            f.write("\n")
        f.write("Release time\n")
        for i in range(self.n):
            f.write(str(self.R[i])+"\t")
        f.write("\n")
        f.write("SSD\n")
        for i in range(self.m):
            f.write("M"+str(i)+"\n")
            for j in range(self.n):
                for k in range(self.n):
                    f.write(str(self.S[i][j][k])+"\t")
                f.write("\n")
        f.close()

    def lower_bound(self):
        """Computes the lower bound of maximal completion time of the instance 
        by dividing the sum of minimal completion time between job pairs on the number of machines

        Returns:
            int: Lower Bound of maximal completion time
        """
        # Preparing ranges
        M = range(self.m)
        E = range(self.n)
        # Compute lower bound
        sum_j = 0
        all_max_r_j = 0
        for j in E:
            min_j = None
            min_r_j = None
            for k in M:
                for i in E:  # (t for t in E if t != j ):
                    if min_j is None or self.P[j][k] + self.S[k][i][j] < min_j:
                        min_j = self.P[j][k] + self.S[k][i][j]
                    if min_r_j is None or self.P[j][k] + self.S[k][i][j] < min_r_j:
                        min_r_j = self.P[j][k] + self.S[k][i][j]
            sum_j += min_j
            all_max_r_j = max(all_max_r_j, min_r_j)

        lb1 = sum_j / self.m
        LB = max(lb1, all_max_r_j)

        return LB

@dataclass
class RmriSijkCmax_Solution(ParallelMachines.ParallelSolution):

    def __init__(self, instance: RmriSijkCmax_Instance = None, configuration: list[ParallelMachines.Machine] = None, objective_value: int = 0):
        """Constructor of RmSijkCmax_Solution

        Args:
            instance (RmriSijkCmax_Instance, optional): Instance to be solved by the solution. Defaults to None.
            configuration (list[ParallelMachines.Machine], optional): list of machines of the instance. Defaults to None.
            objective_value (int, optional): initial objective value of the solution. Defaults to 0.
        """
        self.instance = instance
        if configuration is None:
            self.configuration = []
            for i in range(instance.m):
                machine = ParallelMachines.Machine(i, 0, -1, [])
                self.configuration.append(machine)
        else:
            self.configuration = configuration
        self.objective_value = 0

    def __str__(self):
        return "Cmax : " + str(self.objective_value) + "\n" + "Machine_ID | Job_schedule (job_id , start_time , completion_time) | Completion_time\n" + "\n".join(map(str, self.configuration))

    def copy(self):
        copy_machines = []
        for m in self.configuration:
            copy_machines.append(m.copy())

        copy_solution = RmriSijkCmax_Solution(self.instance)
        for i in range(self.instance.m):
            copy_solution.configuration[i] = copy_machines[i]
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
        configuration_ = []
        for i in range(2, len(content)):
            line_content = content[i].split('|')
            configuration_.append(ParallelMachines.Machine(int(line_content[0]), int(line_content[2]), job_schedule=[ParallelMachines.Job(
                int(j[0]), int(j[1]), int(j[2])) for j in [job.strip()[1:len(job.strip())-1].split(',') for job in line_content[1].split(':')]]))
        solution = cls(objective_value=objective_value_,
                       configuration=configuration_)
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
                for i in range(len(self.configuration)):
                    ticks.append(10*(i+1) + 5)
                    ticks_labels.append(str(i+1))

                gnt.set_yticks(ticks)
                # Labelling tickes of y-axis
                gnt.set_yticklabels(ticks_labels)

                # Setting graph attribute
                gnt.grid(True)

                for j in range(len(self.configuration)):
                    schedule = self.configuration[j].job_schedule
                    prev = -1
                    prevEndTime = 0
                    for element in schedule:
                        job_index, startTime, endTime = element
                        if prevEndTime < startTime:
                            # Idle Time
                            gnt.broken_barh(
                                [(prevEndTime, startTime - prevEndTime)], ((j+1) * 10, 9), facecolors=('tab:gray'))
                        if prev != -1:
                            # Setup Time
                            gnt.broken_barh([(startTime, self.instance.S[j][prev][job_index])], ((
                                j+1) * 10, 9), facecolors=('tab:orange'))
                            # Processing Time
                            gnt.broken_barh([(startTime + self.instance.S[j][prev][job_index],
                                            self.instance.P[job_index][j])], ((j+1) * 10, 9), facecolors=('tab:blue'))
                        else:
                            gnt.broken_barh([(startTime, self.instance.P[job_index][j])], ((
                                j+1) * 10, 9), facecolors=('tab:blue'))
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
        for machine in self.configuration:
            prev_job = None
            ci, setup_time, expected_start_time = 0, 0, 0
            for i, element in enumerate(machine.job_schedule):
                job, startTime, endTime = element
                # Test End Time + start Time
                if prev_job is None:
                    setup_time = self.instance.S[machine.machine_num][job][job]
                    expected_start_time = self.instance.R[job]
                else:
                    setup_time = self.instance.S[machine.machine_num][prev_job][job]
                    expected_start_time = max(ci,self.instance.R[job])

                proc_time = self.instance.P[job][machine.machine_num]
                ci = expected_start_time + proc_time + setup_time

                if startTime != expected_start_time or endTime != ci:
                    print(f'## Error: in machine {machine.machine_num}' +
                          f' found {element} expected {job,expected_start_time, ci}')
                    is_valid = False
                set_jobs.add(job)
                prev_job = job

        is_valid &= len(set_jobs) == self.instance.n
        return is_valid

