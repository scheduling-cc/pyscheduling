import json
import random
from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

import pyscheduling_cc.Problem as Problem

Job = namedtuple('Job', ['id', 'start_time', 'end_time'])


class GenerationProtocol(Enum):
    BASE = 1


class GenerationLaw(Enum):
    UNIFORM = 1
    NORMAL = 2


@dataclass
class SingleInstance(Problem.Instance):

    n: int  # n : Number of jobs

    @classmethod
    @abstractmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            ParallelInstance:

        """
        pass

    @classmethod
    @abstractmethod
    def generate_random(cls, protocol: str = None):
        """Generate a random instance according to a predefined protocol

        Args:
            protocol (string): represents the protocol used to generate the instance

        Returns:
            ParallelInstance:
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
            startIndex (int): Index from where starts the processing time table

        Returns:
           (list[int],int): (Table of processing time, index of the next section of the instance)
        """
        i = startIndex + 1
        line = content[i].strip().split('\t')
        P = []  # Table : Processing time of job i
        for j in line:
            P.append(int(j))
        return (P, i+1)

    def read_W(self, content: list[str], startIndex: int):
        """Read the Processing time matrix from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the jobs weights table

        Returns:
           (list[int],int): (Table of jobs weights, index of the next section of the instance)
        """
        i = startIndex + 1
        line = content[i].strip().split('\t')
        W = []  # Table : Processing time of job i
        for j in line:
            W.append(int(j))
        return (W, i+1)

    def read_R(self, content: list[str], startIndex: int):
        """Read the release time table from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the release time table

        Returns:
           (list[int],int): (Table of release time, index of the next section of the instance)
        """
        i = startIndex + 1
        line = content[i].strip().split('\t')
        ri = []  # Table : Release time of job i
        for j in line:
            ri.append(int(line[j]))
        return (ri, i+1)

    def read_S(self, content: list[str], startIndex: int):
        """Read the Setup time table of matrices from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the Setup time matrix

        Returns:
           (list[list[int]],int): (Matrix of setup time, index of the next section of the instance)
        """
        i = startIndex
        Si = []  # Matrix S_ijk : Setup time between jobs j and k on machine i
        i += 1  # Skip SSD
        for k in range(self.n):
            line = content[i].strip().split('\t')
            Sij = [int(line[j]) for j in range(self.n)]
            Si.append(Sij)
            i += 1
        return (Si, startIndex+1+self.n)

    def read_D(self, content: list[str], startIndex: int):
        """Read the due time table from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the due time table

        Returns:
           (list[int],int): (Table of due time, index of the next section of the instance)
        """
        i = startIndex + 1
        line = content[i].strip().split('\t')
        di = []  # Table : Due time of job i
        for j in line:
            di.append(int(j))
        return (di, i+1)

    def generate_P(self, protocol: GenerationProtocol, law: GenerationLaw, Pmin: int, Pmax: int):
        """Random generation of processing time table

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            Pmin (int): Minimal processing time
            Pmax (int): Maximal processing time

        Returns:
           list[int]: Table of processing time
        """
        P = []
        for j in range(self.n):
            if law.name == "UNIFORM":  # Generate uniformly
                n = int(random.uniform(Pmin, Pmax))
            elif law.name == "NORMAL":  # Use normal law
                value = np.random.normal(0, 1)
                n = int(abs(Pmin+Pmax*value))
                while n < Pmin or n > Pmax:
                    value = np.random.normal(0, 1)
                    n = int(abs(Pmin+Pmax*value))
            P.append(n)

        return P

    def generate_W(self, protocol: GenerationProtocol, law: GenerationLaw, Wmin: int, Wmax: int):
        """Random generation of jobs weights table

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            Wmin (int): Minimal weight
            Wmax (int): Maximal weight

        Returns:
           list[int]: Table of jobs weights
        """
        W = []
        for j in range(self.n):
            if law.name == "UNIFORM":  # Generate uniformly
                n = int(random.uniform(Wmin, Wmax))
            elif law.name == "NORMAL":  # Use normal law
                value = np.random.normal(0, 1)
                n = int(abs(Wmin+Wmax*value))
                while n < Wmin or n > Wmax:
                    value = np.random.normal(0, 1)
                    n = int(abs(Wmin+Wmax*value))
            W.append(n)

        return W
    
    def generate_R(self, protocol: GenerationProtocol, law: GenerationLaw, PJobs: list[float], Pmin: int, Pmax: int, alpha: float):
        """Random generation of release time table

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            PJobs (list[float]): Table of processing time
            Pmin (int): Minimal processing time
            Pmax (int): Maximal processing time
            alpha (float): release time factor

        Returns:
            list[int]: release time table
        """
        ri = []
        for j in range(self.n):
            if law.name == "UNIFORM":  # Generate uniformly
                n = int(random.uniform(0, alpha * PJobs[j]))

            elif law.name == "NORMAL":  # Use normal law
                value = np.random.normal(0, 1)
                n = int(abs(Pmin+Pmax*value))
                while n < Pmin or n > Pmax:
                    value = np.random.normal(0, 1)
                    n = int(abs(Pmin+Pmax*value))

            ri.append(n)

        return ri

    def generate_S(self, protocol: GenerationProtocol, law: GenerationLaw, PJobs: list[float], gamma: float, Smin: int = 0, Smax: int = 0):
        """Random generation of setup time table of matrices

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            PJobs (list[float]): Table of processing time
            gamma (float): Setup time factor
            Smin (int, optional): Minimal setup time . Defaults to 0.
            Smax (int, optional): Maximal setup time. Defaults to 0.

        Returns:
            list[list[int]]: Setup time matrix
        """
        Si = []
        for j in range(self.n):
            Sij = []
            for k in range(self.n):
                if j == k:
                    Sij.append(0)  # check space values
                else:
                    if law.name == "UNIFORM":  # Use uniform law
                        min_p = min(PJobs[k], PJobs[j])
                        max_p = max(PJobs[k], PJobs[j])
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

        return Si

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
    wiCi_index: list[int] = field(default_factory=list) # this table serves as a cache to save the total cweighted completion time reached after each job in job_schedule

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
        self.wiCi_index = None

    def __str__(self):
        return " : ".join(map(str, [(job.id, job.start_time, job.end_time) for job in self.job_schedule])) + " | " + str(self.objective)

    def __eq__(self, other):
        same_schedule = other.job_schedule == self.job_schedule
        return (same_schedule)

    def copy(self):
        return Machine(self.objective, self.last_job, list(self.job_schedule))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    @staticmethod
    def fromDict(machine_dict):
        return Machine(machine_dict["objective"], machine_dict["last_job"], machine_dict["job_schedule"])

    def total_weighted_completion_time(self, instance: SingleInstance, startIndex: int = 0):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job and returns the final completion_time,
        works with both RmSijkCmax and RmriSijkCmax problems

        Args:
            instance (ParallelInstance): The instance associated to the machine
            startIndex (int) : The job index the function starts operating from

        Returns:
            int: objective of the machine
        """
        if len(self.job_schedule) > 0:
            job_schedule_len = len(self.job_schedule)
            if self.wiCi_index is None : # Iniates wiCi_index to the size of job_schedule
                self.wiCi_index = [-1] * job_schedule_len
                startIndex = 0
            if len(self.wiCi_index) != job_schedule_len:
                if startIndex == 0: # If the size is different and no startIndex has been passed, it means a lot of changes has occured as wiCi_index needs to be reinitialized
                    self.wiCi_index = [-1] * job_schedule_len
                else: # Insert an element in wiCi_index corresponding to the position where a new job has been inserted
                    self.wiCi_index.insert(startIndex,-1) 
            if startIndex > 0: 
                ci = self.job_schedule[startIndex - 1].end_time
                wiCi = self.wiCi_index[startIndex - 1]
            else: 
                ci = 0
                wiCi = 0
            for i in range(startIndex,job_schedule_len):
                job_i = self.job_schedule[i].id

                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job_i])
                else:
                    startTime = ci
                proc_time = instance.P[job_i]
                ci = startTime + proc_time

                self.job_schedule[i] = Job(job_i, startTime, ci)
                wiCi += instance.W[job_i]*ci
                self.wiCi_index[i] = wiCi
        self.objective = wiCi
        return wiCi

    def completion_time_insert(self, job: int, pos: int, instance: SingleInstance):
        """
        Computes the machine's completion time if we insert "job" at "pos" in the machine's job_schedule
        Args:
            job (int): id of the inserted job
            pos (int): position where the job is inserted in the machine
            instance (ParallelInstance): the current problem instance
        Returns:
            ci (int) : completion time
        """
        if pos > 0:
            c_prev = self.job_schedule[pos - 1].end_time
            if hasattr(instance, 'R'):
                release_time = max(instance.R[job] - c_prev, 0)
            else:
                release_time = 0 
            ci = c_prev + release_time + instance.P[job]
            wiCi = self.wiCi_index[pos -1]+instance.W[job]*ci
        else: 
            ci = instance.P[job]
            wiCi = instance.W[job]*instance.P[job]
        for i in range(pos, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            if hasattr(instance, 'R'):
                startTime = max(ci, instance.R[job_i])
            else:
                startTime = ci
            proc_time = instance.P[job_i]
            ci = startTime + proc_time
            wiCi += instance.W[job_i]*ci

        return wiCi

    def completion_time_remove_insert(self, pos_remove: int, job: int, pos_insert: int, instance:  SingleInstance):
        """
        Computes the machine's completion time if we remove job at position "pos_remove" 
        and insert "job" at "pos" in the machine's job_schedule
        Args:
            pos_remove (int): position of the job to be removed
            job (int): id of the inserted job
            pos_insert (int): position where the job is inserted in the machine
            instance (ParallelInstance): the current problem instance
        Returns:
            ci (int) : completion time
        """
        first_pos = min(pos_remove, pos_insert)

        ci = 0
        wiCi = 0
        if first_pos > 0:  # There's at least one job in the schedule
            ci = self.job_schedule[first_pos - 1].end_time
            wiCi = self.wiCi_index[first_pos - 1]
        for i in range(first_pos, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            # If job needs to be inserted to position i
            if i == pos_insert:
                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job])
                else:
                    startTime = ci
                proc_time = instance.P[job]
                ci = startTime + proc_time
                wiCi += instance.W[job]*ci

            # If the job_i is not the one to be removed
            if i != pos_remove:
                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job_i])
                else:
                    startTime = ci
                proc_time = instance.P[job_i]
                ci = startTime + proc_time
                wiCi += instance.W[job_i]*ci

        return wiCi

    def completion_time_swap(self, pos_i: int, pos_j: int, instance: SingleInstance):
        """
        Computes the machine's completion time if we insert swap jobs at position "pos_i" and "pos_j"
        in the machine's job_schedule
        Args:
            pos_i (int): position of the first job to be swapped
            pos_j (int): position of the second job to be swapped
            instance (ParallelInstance): the current problem instance
        Returns:
            ci (int) : completion time
        """
        first_pos = min(pos_i, pos_j)

        ci = 0
        wiCi = 0
        if first_pos > 0:  # There's at least one job in the schedule
            ci = self.job_schedule[first_pos - 1].end_time
            wiCi = self.wiCi_index[first_pos - 1]

        for i in range(first_pos, len(self.job_schedule)):

            if i == pos_i:  # We take pos_j
                job_i = self.job_schedule[pos_j][0]  # (Id, startTime, endTime)
            elif i == pos_j:  # We take pos_i
                job_i = self.job_schedule[pos_i][0]
            else:
                job_i = self.job_schedule[i][0]  # Id of job in position i

            if hasattr(instance, 'R'):
                startTime = max(ci, instance.R[job_i])
            else:
                startTime = ci
            proc_time = instance.P[job_i]
            ci = startTime + proc_time
            wiCi += instance.W[job_i]*ci

        return wiCi


@dataclass
class SingleSolution(Problem.Solution):

    machine: Machine

    def __init__(self, instance: SingleInstance):
        """Constructor of SingleSolution

        Args:
            instance (SingleInstance, optional): Instance to be solved by the solution.
        """
        self.instance = instance
        self.machine = Machine(0, -1, [])
        self.objective_value = 0

    def __str__(self):
        return "Objective : " + str(self.objective_value) + "\n" + "Job_schedule (job_id , start_time , completion_time) | objective\n" + self.machine.__str__()

    def copy(self):
        copy_solution = SingleSolution(self.instance)
        for i in range(self.instance.m):
            copy_solution.machine = self.machine.copy()
        copy_solution.objective_value = self.objective_value
        return copy_solution

    def wiCi(self):
        """Sets the job_schedule of every machine associated to the solution and sets the objective_value of the solution to Cmax
            which equals to the maximal completion time of every machine
        """
        if self.instance != None:
                self.machine.total_weighted_completion_time(self.instance)
        self.objective_value = self.machine.objective

    def fix_objective(self):
        """Sets the objective_value of the solution to Cmax
            which equals to the maximal completion time of every machine
        """
        self.objective_value = self.machine.objective

    @classmethod
    @abstractmethod
    def read_txt(cls, path: Path):
        """Read a solution from a txt file

        Args:
            path (Path): path to the solution's txt file of type Path from pathlib

        Returns:
            ParallelSolution:
        """
        pass

    def to_txt(self, path: Path) -> None:
        """Export the solution to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(self.__str__())
        f.close()

    @abstractmethod
    def plot(self) -> None:
        """Plot the solution in an appropriate diagram"""
        pass

