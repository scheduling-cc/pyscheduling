import json
import sys
import random
from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pyscheduling.Problem as RootProblem

Job = namedtuple('Job', ['id', 'start_time', 'end_time'])


class GenerationProtocol(Enum):
    VALLADA = 1


class GenerationLaw(Enum):
    UNIFORM = 1
    NORMAL = 2

@dataclass
class FlowShopInstance(RootProblem.Instance):

    n: int  # n : Number of jobs
    m: int  # m : Number of machines

    @classmethod
    @abstractmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            FlowShopInstance:

        """
        pass

    @classmethod
    @abstractmethod
    def generate_random(cls, protocol: str = None):
        """Generate a random instance according to a predefined protocol

        Args:
            protocol (string): represents the protocol used to generate the instance

        Returns:
            FlowShopInstance:
        """
        pass

    @abstractmethod
    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        pass

    def read_P(self, content: list[str], startIndex: int):
        """Read the Processing time matrix from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the processing time matrix

        Returns:
           (list[list[int]],int): (Matrix of processing time, index of the next section of the instance)
        """
        P = []  # Matrix P_jk : Execution time of job j on machine k
        i = startIndex
        for _ in range(self.n):
            ligne = content[i].strip().split('\t')
            P_k = [int(ligne[j]) for j in range(1, self.m*2, 2)]
            P.append(P_k)
            i += 1
        return (P, i)

    def read_R(self, content: list[str], startIndex: int):
        """Read the release time table from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the release time table

        Returns:
           (list[int],int): (Table of release time, index of the next section of the instance)
        """
        i = startIndex + 1
        ligne = content[i].strip().split('\t')
        ri = []  # Table : Release time of job i
        for j in range(1, len(ligne), 2):
            ri.append(int(ligne[j]))
        return (ri, i+1)

    def read_S(self, content: list[str], startIndex: int):
        """Read the Setup time table of matrices from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the Setup time table of matrices

        Returns:
           (list[list[list[int]]],int): (Table of matrices of setup time, index of the next section of the instance)
        """
        i = startIndex
        S = []  # Table of Matrix S_ijk : Setup time between jobs j and k on machine i
        i += 1  # Skip SSD
        endIndex = startIndex+1+self.n*self.m+self.m
        while i != endIndex:
            i = i+1  # Skip Mk
            Si = []
            for k in range(self.n):
                ligne = content[i].strip().split('\t')
                Sij = [int(ligne[j]) for j in range(self.n)]
                Si.append(Sij)
                i += 1
            S.append(Si)
        return (S, i)

    def read_D(self, content: list[str], startIndex: int):
        """Read the due time table from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the due time table

        Returns:
           (list[int],int): (Table of due time, index of the next section of the instance)
        """
        i = startIndex + 1
        ligne = content[i].strip().split('\t')
        di = []  # Table : Due time of job i
        for j in range(1, len(ligne), 2):
            di.append(int(ligne[j]))
        return (di, i+1)

    def generate_P(self, protocol: GenerationProtocol, law: GenerationLaw, Pmin: int, Pmax: int):
        """Random generation of processing time matrix

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            Pmin (int): Minimal processing time
            Pmax (int): Maximal processing time

        Returns:
           list[list[int]]: Matrix of processing time
        """
        P = []
        for j in range(self.n):
            Pj = []
            for i in range(self.m):
                if law.name == "UNIFORM":  # Generate uniformly
                    n = int(random.uniform(Pmin, Pmax))
                elif law.name == "NORMAL":  # Use normal law
                    value = np.random.normal(0, 1)
                    n = int(abs(Pmin+Pmax*value))
                    while n < Pmin or n > Pmax:
                        value = np.random.normal(0, 1)
                        n = int(abs(Pmin+Pmax*value))
                Pj.append(n)
            P.append(Pj)

        return P

    def generate_R(self, protocol: GenerationProtocol, law: GenerationLaw, PJobs: list[list[float]], Pmin: int, Pmax: int, alpha: float):
        """Random generation of release time table

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            PJobs (list[list[float]]): Matrix of processing time
            Pmin (int): Minimal processing time
            Pmax (int): Maximal processing time
            alpha (float): release time factor

        Returns:
            list[int]: release time table
        """
        ri = []
        for j in range(self.n):
            sum_p = sum(PJobs[j])
            if law.name == "UNIFORM":  # Generate uniformly
                n = int(random.uniform(0, alpha * (sum_p / self.m)))

            elif law.name == "NORMAL":  # Use normal law
                value = np.random.normal(0, 1)
                n = int(abs(Pmin+Pmax*value))
                while n < Pmin or n > Pmax:
                    value = np.random.normal(0, 1)
                    n = int(abs(Pmin+Pmax*value))

            ri.append(n)

        return ri

    def generate_S(self, protocol: GenerationProtocol, law: GenerationLaw, PJobs: list[list[float]], gamma: float, Smin: int = 0, Smax: int = 0):
        """Random generation of setup time table of matrices

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            PJobs (list[list[float]]): Matrix of processing time
            gamma (float): Setup time factor
            Smin (int, optional): Minimal setup time . Defaults to 0.
            Smax (int, optional): Maximal setup time. Defaults to 0.

        Returns:
            list[list[list[int]]]: Setup time table of matrix
        """
        S = []
        for i in range(self.m):
            Si = []
            for j in range(self.n):
                Sij = []
                for k in range(self.n):
                    if j == k:
                        Sij.append(0)  # check space values
                    else:
                        if law.name == "UNIFORM":  # Use uniform law
                            min_p = min(PJobs[k][i], PJobs[j][i])
                            max_p = max(PJobs[k][i], PJobs[j][i])
                            Smin = int(gamma * min_p)
                            Smax = int(gamma * max_p)
                            Sij.append(int(random.uniform(Smin, Smax)))

                        elif law.name == "NORMAL":  # Use normal law
                            value = np.random.normal(0, 1)
                            setup = int(abs(Smin+Smax*value))
                            while setup < Smin or setup > Smax:
                                value = np.random.normal(0, 1)
                                setup = int(abs(Smin+Smax*value))
                            Sij.append(setup)
                Si.append(Sij)
            S.append(Si)

        return S

    def generate_D(self, protocol: GenerationProtocol, law: GenerationLaw, Pmin, Pmax):
        """Random generation of due time table

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            Pmin (int): Minimal processing time
            Pmax (int): Maximal processing time

        Returns:
            list[int]: due time table
        """
        pass


@dataclass
class Machine:
    
    objective: int = 0
    last_job: int = -1
    job_schedule: list[Job] = field(default_factory=list)
    
    
    def __init__(self, objective: int = 0, last_job: int = -1, job_schedule: list[Job] = None) -> None:
        """Constructor of Machine

        Args:
            objective (int, optional): completion time of the last job of the machine. Defaults to 0.
            last_job (int, optional): ID of the last job set on the machine. Defaults to -1.
            job_schedule (list[Job], optional): list of Jobs scheduled on the machine in the exact given sequence. Defaults to None.
        """
        self.objective = objective
        self.last_job = last_job
        if job_schedule is None:
            self.job_schedule = []
        else:
            self.job_schedule = job_schedule

    def __str__(self):
        return " : ".join(map(str, [(job.id, job.start_time, job.end_time) for job in self.job_schedule])) + " | " + str(self.objective)

    def __eq__(self, other):
        same_schedule = other.job_schedule == self.job_schedule
        return (same_schedule)

    def copy(self):
        if self.wiCi_index is None and self.wiTi_index is None:
            return Machine(self.objective, self.last_job, list(self.job_schedule))
        elif self.wiCi_index is None:
            return Machine(self.objective, self.last_job, list(self.job_schedule),wiTi_index=list(self.wiTi_index))
        else: return Machine(self.objective, self.last_job, list(self.job_schedule),wiCi_index=list(self.wiCi_index))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    @staticmethod
    def fromDict(machine_dict):
        return Machine(machine_dict["objective"], machine_dict["last_job"], machine_dict["job_schedule"])

    def completion_time(self, instance : FlowShopInstance, startIndex : int = 0):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job and returns the maximal completion time

        Args:
            instance (SingleInstance): The instance associated to the machine
            startIndex (int) : The job index the function starts operating from

        Returns:
            int: Makespan
        """
        if len(self.job_schedule) > 0:
            job_schedule_len = len(self.job_schedule)
            if startIndex > 0: 
                ci = self.job_schedule[startIndex - 1].end_time
                job_prev_i = self.job_schedule[startIndex - 1].id
            else: 
                ci = 0
                job_prev_i = self.job_schedule[startIndex].id
            for i in range(startIndex,job_schedule_len):
                job_i = self.job_schedule[i].id

                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job_i])
                else:
                    startTime = ci
                if hasattr(instance, 'S'):
                    setupTime = instance.S[job_prev_i][job_i]
                else:
                    setupTime = 0
                proc_time = instance.P[job_i]
                ci = startTime + setupTime + proc_time

                self.job_schedule[i] = Job(job_i, startTime, ci)
                job_prev_i = job_i
        self.objective = ci
        return ci


@dataclass
class FlowShopSolution(RootProblem.Solution):

    machines: list[Machine]
    job_schedule = list[int]

    def __init__(self, instance: FlowShopInstance = None, machines: list[Machine] = None, job_schedule : list[int] = None, objective_value: int = 0):
        """Constructor of RmSijkCmax_Solution

        Args:
            instance (FlowShopInstance, optional): Instance to be solved by the solution. Defaults to None.
            machines (list[Machine], optional): list of machines of the instance. Defaults to None.
            job_schedule (list[int], optional): sequence of scheduled jobs. Defaults to None.
            objective_value (int, optional): initial objective value of the solution. Defaults to 0.
        """
        self.instance = instance
        if machines is None:
            self.machines = []
            for i in range(instance.m):
                machine = Machine(0, -1, [])
                self.machines.append(machine)
        else:
            self.machines = machines
        if job_schedule is None: job_schedule = []
        else: self.job_schedule = job_schedule
        self.objective_value = 0

    def __str__(self):
        return "Objective : " + str(self.objective_value) + "\n" + "Jobs sequence : " + "\t".join(map(str, self.job_schedule)) + "\n" + "Machine_ID | Job_schedule (job_id , start_time , completion_time) | Completion_time\n" + "\n".join(map(str, self.machines))

    def copy(self):
        copy_machines = []
        for m in self.machines:
            copy_machines.append(m.copy())

        copy_solution = FlowShopSolution(self.instance)
        for i in range(self.instance.m):
            copy_solution.machines[i] = copy_machines[i]
        copy_solution.job_schedule = list(self.job_schedule)
        copy_solution.objective_value = self.objective_value
        return copy_solution

    def __lt__(self, other):
        if self.instance.get_objective().value > 0 :
            return self.objective_value < other.objective_value
        else : return other.objective_value < self.objective_value
    
    def init_machines_schedule(self):
        """Fills the job_schedule of every machine from job_schedule of Solution
        """
        for machine in self.machines :
            machine.job_schedule = [Job(job_id,0,0) for job_id in self.job_schedule]
    
    def cmax(self, start_job_index : int = 0):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job
         at every machine and returns the maximal completion time which is the completion time of the last machine

        Args:
            start_job_index (int, optional): starting index to update the job_schedule from for every stage (machine). Defaults to 0.
        """
        if start_job_index == 0 : self.init_machines_schedule()
        elif len(self.job_schedule) != len(self.machines[0].job_schedule):
            for machine in self.machines :
                machine.job_schedule.insert(start_job_index,Job(0,0,0))
        ci = 0
        prev_job = -1
        if start_job_index > 0:
            ci = self.machines[0].job_schedule[start_job_index - 1].end_time
            prev_job = self.machines[0].job_schedule[start_job_index - 1].id
        job_index = start_job_index
        for job_id in self.job_schedule[start_job_index:]:
            if hasattr(self.instance,'S'):
                if prev_job == -1:
                    setupTime = self.instance.S[0][job_id][job_id]
                else:
                    setupTime = self.instance.S[0][prev_job][job_id]
            else: setupTime = 0
            startTime = ci
            ci = startTime + setupTime + self.instance.P[job_id][0]
            self.machines[0].job_schedule[job_index] = (Job(job_id,startTime,ci))
            job_index += 1
            prev_job = job_id
        self.machines[0].objective = ci
        self.machines[0].last_job = job_id

        prev_machine = 0
        for machine_id in range(1,self.instance.m):
            ci = self.machines[prev_machine].job_schedule[0].end_time
            prev_job = -1
            if start_job_index > 0:
                ci = self.machines[machine_id].job_schedule[start_job_index - 1].end_time
                prev_job = self.machines[machine_id].job_schedule[start_job_index - 1].id
            job_index = start_job_index
            for job_id in self.job_schedule[start_job_index:]:
                if hasattr(self.instance,'S'):
                    if prev_job == -1:
                        setupTime = self.instance.S[machine_id][job_id][job_id]
                    else:
                        setupTime = self.instance.S[machine_id][prev_job][job_id]
                else: setupTime = 0
                startTime = max(ci,self.machines[prev_machine].job_schedule[job_index].end_time)
                remaining_setupTime = max(setupTime-(startTime-ci),0)
                ci = startTime + remaining_setupTime + self.instance.P[job_id][machine_id]
                self.machines[machine_id].job_schedule[job_index] = Job(job_id,startTime,ci)
                job_index += 1
                prev_job = job_id
            self.machines[machine_id].objective = ci
            self.machines[machine_id].last_job = job_id

            prev_machine = machine_id

        self.objective_value = self.machines[self.instance.m - 1].objective
    
    def idle_time_cmax_insert_last_pos(self, job_id : int):
        """returns start_time and completion_time of job_id if scheduled at the end of job_schedule
        at every stage (machine)

        Args:
            job_id (int): job to be scheduled at the end

        Returns:
            int, int: start_time of job_id, completion_time of job_id
        """
        ci = 0
        prev_job = -1
        job_schedule_len = len(self.job_schedule)
        if job_schedule_len > 0:
            ci = self.machines[0].job_schedule[job_schedule_len - 1].end_time
            prev_job = self.machines[0].job_schedule[job_schedule_len - 1].id
 

        if hasattr(self.instance,'S'):
            if prev_job == -1:
                setupTime = self.instance.S[0][job_id][job_id]
            else:
                setupTime = self.instance.S[0][prev_job][job_id]
        else: setupTime = 0
        startTime = ci
        new_ci = startTime + setupTime + self.instance.P[job_id][0]

        for machine_id in range(1,self.instance.m):
            ci = new_ci
            prev_job = -1
            if job_schedule_len > 0:
                ci = self.machines[machine_id].job_schedule[job_schedule_len - 1].end_time
                prev_job = self.machines[machine_id].job_schedule[job_schedule_len - 1].id
                if hasattr(self.instance,'S'):
                    if prev_job == -1:
                        setupTime = self.instance.S[machine_id][job_id][job_id]
                    else:
                        setupTime = self.instance.S[machine_id][prev_job][job_id]
                else: setupTime = 0
                startTime = max(ci,new_ci)
                remaining_setupTime = max(setupTime-(startTime-ci),0)
                new_ci = startTime + remaining_setupTime + self.instance.P[job_id][machine_id]

        return startTime,new_ci
    
    def idle_time(self):
        """returns the idle time of the last machine

        Returns:
            int: idle time of the last machine
        """
        last_machine = self.machines[self.instance.m-1]
        idleTime = last_machine.job_schedule[0].start_time
        for job_index in range(len(last_machine.job_schedule)-1):
            idleTime += last_machine.job_schedule[job_index+1].start_time - last_machine.job_schedule[job_index].end_time
        return idleTime
    
    @classmethod
    def read_txt(cls, path: Path):
        """Read a solution from a txt file

        Args:
            path (Path): path to the solution's txt file of type Path from pathlib

        Returns:
            FlowShopSolution:
        """
        f = open(path, "r")
        content = f.read().split('\n')
        objective_value_ = int(content[0].split(':')[1])
        configuration_ = []
        for i in range(2, len(content)):
            line_content = content[i].split('|')
            configuration_.append(Machine(int(line_content[0]), int(line_content[2]), job_schedule=[Job(
                int(j[0]), int(j[1]), int(j[2])) for j in [job.strip()[1:len(job.strip())-1].split(',') for job in line_content[1].split(':')]]))
        solution = cls(objective_value=objective_value_,
                       machines=configuration_)
        return solution

    def to_txt(self, path: Path) -> None:
        """Export the solution to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(self.__str__())
        f.close()

    def plot(self, path: Path = None) -> None:
        """Plot the solution in a gantt diagram"""
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
                for i in range(len(self.machines)):
                    ticks.append(10*(i+1) + 5)
                    ticks_labels.append(str(i+1))

                gnt.set_yticks(ticks)
                # Labelling tickes of y-axis
                gnt.set_yticklabels(ticks_labels)

                # Setting graph attribute
                gnt.grid(True)

                for j in range(len(self.machines)):
                    schedule = self.machines[j].job_schedule
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
        return True


class FS_LocalSearch(RootProblem.LocalSearch):
    pass
    


class NeighbourhoodGeneration():
    pass