import json
import random
import sys
from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pyscheduling.Problem as RootProblem
from matplotlib import pyplot as plt

Job = namedtuple('Job', ['id', 'start_time', 'end_time'])


class GenerationProtocol(Enum):
    BASE = 1


class GenerationLaw(Enum):
    UNIFORM = 1
    NORMAL = 2


@dataclass
class SingleInstance(RootProblem.Instance):

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
            SingleInstance:

        """
        pass

    @classmethod
    @abstractmethod
    def generate_random(cls, protocol: str = None):
        """Generate a random instance according to a predefined protocol

        Args:
            protocol (string): represents the protocol used to generate the instance

        Returns:
            SingleInstance:
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
        """Read the Processing time table from a list of lines extracted from the file of the instance

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
        """Read the Processing time table from a list of lines extracted from the file of the instance

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
            ri.append(int(j))
        return (ri, i+1)

    def read_S(self, content: list[str], startIndex: int):
        """Read the Setup time matrix from a list of lines extracted from the file of the instance

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
        """Random generation of setup time matrix

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

    def generate_D(self, protocol: GenerationProtocol, law: GenerationLaw, PJobs: list[float], Pmin : int, Pmax : int, due_time_factor : float, RJobs : list[float] = None):
        """Random generation of due time table

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            PJobs (list[float]): Table of processing time
            Pmin (int): Minimal processing time
            Pmax (int): Maximal processing time
            fraction (float): due time factor

        Returns:
            list[int]: due time table
        """
        di = []
        sumP = sum(PJobs)
        for j in range(self.n):
            if hasattr(self,'R'): startTime = RJobs[j] + PJobs[j]
            else: startTime = PJobs[j]
            if law.name == "UNIFORM":  # Generate uniformly    
                n = int(random.uniform(startTime, startTime + due_time_factor * sumP))

            elif law.name == "NORMAL":  # Use normal law
                value = np.random.normal(0, 1)
                n = int(abs(Pmin+Pmax*value))
                while n < Pmin or n > Pmax:
                    value = np.random.normal(0, 1)
                    n = int(abs(Pmin+Pmax*value))

            di.append(n)

        return di


@dataclass
class Machine:

    objective: int = 0
    last_job: int = -1
    job_schedule: list[Job] = field(default_factory=list)
    wiCi_index: list[int] = field(default_factory=list) # this table serves as a cache to save the total weighted completion time reached after each job in job_schedule
    wiTi_index: list[int] = field(default_factory=list) # this table serves as a cache to save the total weighted lateness reached after each job in job_schedule
    
    def __init__(self, objective: int = 0, last_job: int = -1, job_schedule: list[Job] = None, wiCi_index : list[int] = None, wiTi_index : list[int] = None) -> None:
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
        self.wiCi_index = wiCi_index
        self.wiTi_index = wiTi_index

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

    def total_weighted_completion_time(self, instance: SingleInstance, startIndex: int = 0):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job and returns the total weighted completion time

        Args:
            instance (SingleInstance): The instance associated to the machine
            startIndex (int) : The job index the function starts operating from

        Returns:
            int: Total weighted lateness
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
                job_prev_i = self.job_schedule[startIndex - 1].id
                wiCi = self.wiCi_index[startIndex - 1]
            else: 
                ci = 0
                job_prev_i = self.job_schedule[startIndex].id
                wiCi = 0
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

                self.job_schedule[i] = Job(job_i, startTime + setupTime, ci)
                wiCi += instance.W[job_i]*ci
                self.wiCi_index[i] = wiCi

                job_prev_i = job_i
        self.objective = wiCi
        return wiCi

    def total_weighted_completion_time_insert(self, job: int, pos: int, instance: SingleInstance):
        """Computes the machine's total wighted completion time if we insert "job" at "pos" in the machine's job_schedule
        
        Args:
            job (int): id of the inserted job
            pos (int): position where the job is inserted in the machine
            instance (SingleInstance): the current problem instance
        Returns:
            int : total weighted completion time
        """
        if pos > 0:
            c_prev = self.job_schedule[pos - 1].end_time
            job_prev_i = self.job_schedule[pos - 1].id
            if hasattr(instance, 'R'):
                release_time = max(instance.R[job] - c_prev, 0)
            else:
                release_time = 0 
            if hasattr(instance, 'S'):
                setupTime = instance.S[job_prev_i][job]
            else:
                setupTime = 0
            ci = c_prev + release_time + setupTime +instance.P[job]
            wiCi = self.wiCi_index[pos -1]+instance.W[job]*ci
        else: 
            ci = instance.S[job][job] + instance.P[job]
            wiCi = instance.W[job]*ci
        job_prev_i = job
        for i in range(pos, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            if hasattr(instance, 'R'):
                startTime = max(ci, instance.R[job_i])
            else:
                startTime = ci
            if hasattr(instance, 'S'):
                setupTime = instance.S[job_prev_i][job_i]
            else:
                setupTime = 0
            proc_time = instance.P[job_i]
            ci = startTime + setupTime +proc_time
            wiCi += instance.W[job_i]*ci

            job_prev_i = job_i

        return wiCi

    def total_weighted_completion_time_remove_insert(self, pos_remove: int, job: int, pos_insert: int, instance:  SingleInstance):
        """Computes the machine's total weighted completion time if we remove job at position "pos_remove" 
        and insert "job" at "pos" in the machine's job_schedule
        
        Args:
            pos_remove (int): position of the job to be removed
            job (int): id of the inserted job
            pos_insert (int): position where the job is inserted in the machine
            instance (SingleInstance): the current problem instance
        Returns:
            int: total weighted completion time
        """
        first_pos = min(pos_remove, pos_insert)

        ci = 0
        wiCi = 0
        job_prev_i=job
        if first_pos > 0:  # There's at least one job in the schedule
            ci = self.job_schedule[first_pos - 1].end_time
            job_prev_i = self.job_schedule[first_pos - 1].id
            wiCi = self.wiCi_index[first_pos - 1]
        for i in range(first_pos, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            # If job needs to be inserted to position i
            if i == pos_insert:
                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job])
                else:
                    startTime = ci
                if hasattr(instance, 'S'):
                    setupTime = instance.S[job_prev_i][job]
                else:
                    setupTime = 0
                proc_time = instance.P[job]
                ci = startTime + setupTime + proc_time
                wiCi += instance.W[job]*ci

            # If the job_i is not the one to be removed
            if i != pos_remove:
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
                wiCi += instance.W[job_i]*ci

            job_prev_i = job_i

        return wiCi

    def total_weighted_completion_time_swap(self, pos_i: int, pos_j: int, instance: SingleInstance):
        """Computes the machine's total weighted completion time if we insert swap jobs at position "pos_i" and "pos_j"
        in the machine's job_schedule
        Args:
            pos_i (int): position of the first job to be swapped
            pos_j (int): position of the second job to be swapped
            instance (SingleInstance): the current problem instance
        Returns:
            int : total weighted completion time
        """
        first_pos = min(pos_i, pos_j)

        ci = 0
        if pos_i == 0: job_prev_i = self.job_schedule[pos_j]
        else: job_prev_i = self.job_schedule[pos_i]
        wiCi = 0
        if first_pos > 0:  # There's at least one job in the schedule
            ci = self.job_schedule[first_pos - 1].end_time
            job_prev_i = self.job_schedule[first_pos - 1].id
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
            if hasattr(instance, 'S'):
                setupTime = instance.S[job_prev_i][job_i]
            else:
                setupTime = 0
            proc_time = instance.P[job_i]
            ci = startTime + setupTime + proc_time
            wiCi += instance.W[job_i]*ci

            job_prev_i = job_i

        return wiCi

    def total_weighted_lateness(self, instance: SingleInstance, startIndex: int = 0):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job and returns the total weighted lateness

        Args:
            instance (SingleInstance): The instance associated to the machine
            startIndex (int) : The job index the function starts operating from

        Returns:
            int: Total weighted lateness
        """
        if len(self.job_schedule) > 0:
            job_schedule_len = len(self.job_schedule)
            if self.wiTi_index is None : # Iniates wiTi_index to the size of job_schedule
                self.wiTi_index = [-1] * job_schedule_len
                startIndex = 0
            if len(self.wiTi_index) != job_schedule_len:
                if startIndex == 0: # If the size is different and no startIndex has been passed, it means a lot of changes has occured as wiTi_index needs to be reinitialized
                    self.wiTi_index = [-1] * job_schedule_len
                else: # Insert an element in wiTi_index corresponding to the position where a new job has been inserted
                    self.wiTi_index.insert(startIndex,-1) 
            if startIndex > 0: 
                ci = self.job_schedule[startIndex - 1].end_time
                job_prev_i = self.job_schedule[startIndex - 1].id
                wiTi = self.wiTi_index[startIndex - 1]
            else: 
                ci = 0
                job_prev_i = self.job_schedule[startIndex].id
                wiTi = 0
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
                wiTi += instance.W[job_i]*max(ci-instance.D[job_i],0)
                self.wiTi_index[i] = wiTi
                job_prev_i = job_i
        self.objective = wiTi
        return wiTi

    def total_weighted_lateness_insert(self, job: int, pos: int, instance: SingleInstance):
        """Computes the machine's total wighted lateness if we insert "job" at "pos" in the machine's job_schedule
        Args:
            job (int): id of the inserted job
            pos (int): position where the job is inserted in the machine
            instance (SingleInstance): the current problem instance
        Returns:
            int : total weighted lateness
        """
        if pos > 0:
            c_prev = self.job_schedule[pos - 1].end_time
            job_prev_i = self.job_schedule[pos - 1].id
            if hasattr(instance, 'R'):
                release_time = max(instance.R[job] - c_prev, 0)
            else:
                release_time = 0 
            if hasattr(instance, 'S'):
                setupTime = instance.S[job_prev_i][job]
            else:
                setupTime = 0
            ci = c_prev + release_time + setupTime +instance.P[job]
            wiTi = self.wiTi_index[pos -1]+instance.W[job]*ci
        else: 
            ci = instance.S[job][job] + instance.P[job]
            wiTi = instance.W[job]*ci
        job_prev_i = job
        for i in range(pos, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            if hasattr(instance, 'R'):
                startTime = max(ci, instance.R[job_i])
            else:
                startTime = ci
            if hasattr(instance, 'S'):
                setupTime = instance.S[job_prev_i][job_i]
            else:
                setupTime = 0
            proc_time = instance.P[job_i]
            ci = startTime + setupTime +proc_time
            wiTi += instance.W[job_i]*max(ci-instance.D[job_i],0)

            job_prev_i = job_i

        return wiTi

    def total_weighted_lateness_remove_insert(self, pos_remove: int, job: int, pos_insert: int, instance:  SingleInstance):
        """Computes the machine's total weighted lateness if we remove job at position "pos_remove" 
        and insert "job" at "pos" in the machine's job_schedule
        Args:
            pos_remove (int): position of the job to be removed
            job (int): id of the inserted job
            pos_insert (int): position where the job is inserted in the machine
            instance (SingleInstance): the current problem instance
        Returns:
            int: total weighted lateness
        """
        first_pos = min(pos_remove, pos_insert)

        ci = 0
        wiTi = 0
        job_prev_i=job
        if first_pos > 0:  # There's at least one job in the schedule
            ci = self.job_schedule[first_pos - 1].end_time
            job_prev_i = self.job_schedule[first_pos - 1].id
            wiTi = self.wiTi_index[first_pos - 1]
        for i in range(first_pos, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            # If job needs to be inserted to position i
            if i == pos_insert:
                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job])
                else:
                    startTime = ci
                if hasattr(instance, 'S'):
                    setupTime = instance.S[job_prev_i][job]
                else:
                    setupTime = 0
                proc_time = instance.P[job]
                ci = startTime + setupTime + proc_time
                wiTi += instance.W[job_i]*max(ci-instance.D[job_i],0)

            # If the job_i is not the one to be removed
            if i != pos_remove:
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
                wiTi += instance.W[job_i]*max(ci-instance.D[job_i],0)

            job_prev_i = job_i

        return wiTi

    def total_weighted_lateness_swap(self, pos_i: int, pos_j: int, instance: SingleInstance):
        """Computes the machine's total weighted lateness if we insert swap jobs at position "pos_i" and "pos_j"
        in the machine's job_schedule
        Args:
            pos_i (int): position of the first job to be swapped
            pos_j (int): position of the second job to be swapped
            instance (SingleInstance): the current problem instance
        Returns:
            int : total weighted lateness
        """
        first_pos = min(pos_i, pos_j)

        ci = 0
        if pos_i == 0: job_prev_i = self.job_schedule[pos_j].id
        else: job_prev_i = self.job_schedule[pos_i].id
        wiTi = 0
        if first_pos > 0:  # There's at least one job in the schedule
            ci = self.job_schedule[first_pos - 1].end_time
            job_prev_i = self.job_schedule[first_pos - 1].id
            wiTi = self.wiTi_index[first_pos - 1]

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
            if hasattr(instance, 'S'):
                setupTime = instance.S[job_prev_i][job_i]
            else:
                setupTime = 0
            proc_time = instance.P[job_i]
            ci = startTime + setupTime + proc_time
            wiTi += instance.W[job_i]*max(ci-instance.D[job_i],0)

            job_prev_i = job_i

        return wiTi

    def completion_time(self, instance : SingleInstance, startIndex : int = 0):
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

    def completion_time_insert(self, job: int, pos: int, instance: SingleInstance):
        """Computes the machine's completion time if we insert "job" at "pos" in the machine's job_schedule

        Args:
            job (int): id of the inserted job
            pos (int): position where the job is inserted in the machine
            instance (SingleInstance): the current problem instance
        Returns:
            int : completion time
        """
        if pos > 0:
            c_prev = self.job_schedule[pos - 1].end_time
            job_prev_i = self.job_schedule[pos - 1].id
            if hasattr(instance, 'R'):
                release_time = max(instance.R[job] - c_prev, 0)
            else:
                release_time = 0 
            if hasattr(instance, 'S'):
                setupTime = instance.S[job_prev_i][job]
            else:
                setupTime = 0
            ci = c_prev + release_time + setupTime +instance.P[job]
        else: 
            ci = instance.S[job][job] + instance.P[job]
        job_prev_i = job
        for i in range(pos, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            if hasattr(instance, 'R'):
                startTime = max(ci, instance.R[job_i])
            else:
                startTime = ci
            if hasattr(instance, 'S'):
                setupTime = instance.S[job_prev_i][job_i]
            else:
                setupTime = 0
            proc_time = instance.P[job_i]
            ci = startTime + setupTime +proc_time

            job_prev_i = job_i

        return ci

    def completion_time_remove_insert(self, pos_remove: int, job: int, pos_insert: int, instance:  SingleInstance):
        """Computes the machine's completion time if we remove job at position "pos_remove" 
        and insert "job" at "pos" in the machine's job_schedule
        
        Args:
            pos_remove (int): position of the job to be removed
            job (int): id of the inserted job
            pos_insert (int): position where the job is inserted in the machine
            instance (SingleInstance): the current problem instance
        Returns:
            int: Completion time
        """
        first_pos = min(pos_remove, pos_insert)

        ci = 0

        job_prev_i=job
        if first_pos > 0:  # There's at least one job in the schedule
            ci = self.job_schedule[first_pos - 1].end_time
            job_prev_i = self.job_schedule[first_pos - 1].id
        for i in range(first_pos, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            # If job needs to be inserted to position i
            if i == pos_insert:
                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job])
                else:
                    startTime = ci
                if hasattr(instance, 'S'):
                    setupTime = instance.S[job_prev_i][job]
                else:
                    setupTime = 0
                proc_time = instance.P[job]
                ci = startTime + setupTime + proc_time

            # If the job_i is not the one to be removed
            if i != pos_remove:
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

            job_prev_i = job_i

        return ci

    def completion_time_swap(self, pos_i: int, pos_j: int, instance: SingleInstance):
        """Computes the machine's completion time if we insert swap jobs at position "pos_i" and "pos_j"
        in the machine's job_schedule
        
        Args:
            pos_i (int): position of the first job to be swapped
            pos_j (int): position of the second job to be swapped
            instance (SingleInstance): the current problem instance
        Returns:
            int : completion time
        """
        first_pos = min(pos_i, pos_j)

        ci = 0
        if pos_i == 0: job_prev_i = self.job_schedule[pos_j].id
        else: job_prev_i = self.job_schedule[pos_i].id
        if first_pos > 0:  # There's at least one job in the schedule
            ci = self.job_schedule[first_pos - 1].end_time
            job_prev_i = self.job_schedule[first_pos - 1].id

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
            if hasattr(instance, 'S'):
                setupTime = instance.S[job_prev_i][job_i]
            else:
                setupTime = 0
            proc_time = instance.P[job_i]
            ci = startTime + setupTime + proc_time

            job_prev_i = job_i

        return ci



@dataclass
class SingleSolution(RootProblem.Solution):

    machine: Machine

    def __init__(self, instance: SingleInstance, machine : Machine = None, objective_value: int = 0):
        """Constructor of SingleSolution

        Args:
            instance (SingleInstance, optional): Instance to be solved by the solution.
        """
        self.instance = instance
        if machine is None:
            self.machine = Machine(0, -1, [])
        else:
            self.machine = machine
        self.objective_value = objective_value

    def __str__(self):
        return "Objective : " + str(self.objective_value) + "\n" + "Job_schedule (job_id , start_time , completion_time) | objective\n" + self.machine.__str__()

    def copy(self):
        copy_solution = SingleSolution(self.instance)
        copy_solution.machine = self.machine.copy()
        copy_solution.objective_value = self.objective_value
        return copy_solution

    def __lt__(self, other):
        if self.instance.get_objective().value > 0 :
            return self.objective_value < other.objective_value
        else : return other.objective_value < self.objective_value

    def wiCi(self):
        """Sets the job_schedule of the machine and affects the total weighted completion time to the objective_value attribute
        """
        if self.instance != None:
                self.machine.total_weighted_completion_time(self.instance)
        self.objective_value = self.machine.objective

    def wiTi(self):
        """Sets the job_schedule of the machine and affects the total weighted lateness to the objective_value attribute
        """
        if self.instance != None:
                self.machine.total_weighted_lateness(self.instance)
        self.objective_value = self.machine.objective

    def Cmax(self):
        """Sets the job_schedule of the machine and affects the makespan to the objective_value attribute
        """
        if self.instance != None:
                self.machine.completion_time(self.instance)
        self.objective_value = self.machine.objective

    def fix_objective(self):
        """Sets the objective_value attribute of the solution to the objective attribute of the machine
        """
        self.objective_value = self.machine.objective

    @classmethod
    def read_txt(cls, path: Path):
        """Read a solution from a txt file

        Args:
            path (Path): path to the solution's txt file of type Path from pathlib

        Returns:
            SingleSolution:
        """
        f = open(path, "r")
        content = f.read().split('\n')
        objective_value_ = int(content[0].split(':')[1])
        line_content = content[2].split('|')
        machine = Machine(int(line_content[1]), job_schedule=[Job(
                int(j[0]), int(j[1]), int(j[2])) for j in [job.strip()[1:len(job.strip())-1].split(',') for job in line_content[0].split(':')]])
        solution = cls(objective_value=objective_value_,
                       machine=machine)
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
                        if hasattr(self.instance, 'S'):
                            setupTime = self.instance.S[prev][job_index]
                        else:
                            setupTime = 0
                            # Setup Time
                        gnt.broken_barh([(startTime, setupTime)], (
                            10, 9), facecolors=('tab:orange'))
                        # Processing Time
                        gnt.broken_barh([(startTime + setupTime,
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
        prev_job = None
        ci, setup_time, expected_start_time = 0, 0, 0
        for i, element in enumerate(self.machine.job_schedule):
            job, startTime, endTime = element
            # Test End Time + start Time
            if hasattr(self.instance,'R'):
                expected_start_time = max(self.instance.R[job],ci)
            else:
                expected_start_time = ci
            if hasattr(self.instance,'S'):
                if prev_job is None:
                    setup_time = self.instance.S[job][job]
                else:
                    setup_time = self.instance.S[prev_job][job]
            else: setup_time = 0

            proc_time = self.instance.P[job]
            ci = expected_start_time + proc_time + setup_time

            if startTime != expected_start_time or endTime != ci:
                print(f'## Error:  found {element} expected {job,expected_start_time, ci}')
                is_valid = False
            set_jobs.add(job)

            prev_job = job

        is_valid &= len(set_jobs) == self.instance.n
        return is_valid

class SM_LocalSearch(RootProblem.LocalSearch):

    @staticmethod
    def _intra_insertion(solution : SingleSolution, objective : RootProblem.Objective):
        """Iterates through the job schedule and try to reschedule every job at a better position to improve the solution

        Args:
            solution (SingleSolution): solution to improve
            objective (RootProblem.Objective): objective to consider

        Returns:
            SingleSolution: improved solution
        """
        if objective == RootProblem.Objective.wiCi:
            fix_machine = solution.machine.total_weighted_completion_time
            remove_insert = solution.machine.total_weighted_completion_time_remove_insert
        elif objective == RootProblem.Objective.wiTi:
            fix_machine = solution.machine.total_weighted_lateness
            remove_insert = solution.machine.total_weighted_lateness_remove_insert
        elif objective == RootProblem.Objective.Cmax:
            fix_machine = solution.machine.completion_time
            remove_insert = solution.machine.completion_time_remove_insert
        for pos in range(len(solution.machine.job_schedule)):
            job = solution.machine.job_schedule[pos]
            objective = solution.machine.objective
            taken_pos = pos
            for new_pos in range(len(solution.machine.job_schedule)):
                if(pos != new_pos):
                    new_objective = remove_insert(pos,job.id,new_pos,solution.instance)
                    if new_objective < objective: 
                        taken_pos = new_pos
                        objective = new_objective
            if taken_pos != pos:
                solution.machine.job_schedule.pop(pos)
                solution.machine.job_schedule.insert(taken_pos,job)
                fix_machine(solution.instance,min(taken_pos,pos))
        solution.fix_objective()
        return solution

    @staticmethod
    def _swap(solution : SingleSolution, objective : RootProblem.Objective):
        """Iterates through the job schedule and choose the best swap between 2 jobs to improve the solution

        Args:
            solution (SingleSolution): solution to improve
            objective (RootProblem.Objective): objective to consider

        Returns:
            SingleSolution: improved solution
        """
        if objective == RootProblem.Objective.wiCi:
            set_objective = solution.wiCi
            swap = solution.machine.total_weighted_completion_time_swap
        elif objective == RootProblem.Objective.wiTi:
            set_objective = solution.wiTi
            swap = solution.machine.total_weighted_lateness_swap
        elif objective == RootProblem.Objective.Cmax:
            set_objective = solution.Cmax
            swap = solution.machine.completion_time_swap

        job_schedule_len = len(solution.machine.job_schedule)
        move = None
        for i in range(0, job_schedule_len):
            for j in range(i+1, job_schedule_len):
                new_ci = swap(i,j,solution.instance)
                if new_ci < solution.machine.objective:
                    if not move:
                        move = (i, j, new_ci)
                    elif new_ci < move[2]:
                        move = (i, j, new_ci)

        if move:
            solution.machine.job_schedule[move[0]], solution.machine.job_schedule[move[1]
            ] = solution.machine.job_schedule[move[1]], solution.machine.job_schedule[move[0]]
            solution.machine.objective = move[2]
            set_objective()
        return solution

    def improve(self, solution: SingleSolution, objective : RootProblem.Objective) -> SingleSolution:
        """Improves a solution by iteratively calling local search operators

        Args:
            solution (Solution): current solution

        Returns:
            Solution: improved solution
        """
        curr_sol = solution.copy() if self.copy_solution else solution
        for method in self.methods:
            curr_sol = method(curr_sol,objective)

        return curr_sol


class NeighbourhoodGeneration():
    @staticmethod
    def random_swap(solution: SingleSolution, objective : RootProblem.Objective, force_improve: bool = True):
        """Performs a random swap between 2 jobs

        Args:
            solution (SingleSolution): Solution to be improved
            objective (RootProblem.Objective) : objective to consider
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.

        Returns:
            SingleSolution: New solution
        """

        if objective == RootProblem.Objective.wiCi:
            fix_machine = solution.machine.total_weighted_completion_time
            swap = solution.machine.total_weighted_completion_time_swap
        elif objective == RootProblem.Objective.wiTi:
            fix_machine = solution.machine.total_weighted_lateness
            swap = solution.machine.total_weighted_lateness_swap
        elif objective == RootProblem.Objective.Cmax:
            fix_machine = solution.machine.completion_time
            swap = solution.machine.completion_time_swap

        machine_schedule = solution.machine.job_schedule
        machine_schedule_len = len(machine_schedule)

        old_ci = solution.machine.objective

        random_job_index = random.randrange(machine_schedule_len)
        other_job_index = random.randrange(machine_schedule_len)

        while other_job_index == random_job_index:
            other_job_index = random.randrange(machine_schedule_len)

        new_ci = swap(
            random_job_index, other_job_index, solution.instance)

        # Apply the move
        if not force_improve or (new_ci <= old_ci):
            machine_schedule[random_job_index], machine_schedule[
                other_job_index] = machine_schedule[
                    other_job_index], machine_schedule[random_job_index]

            fix_machine(solution.instance,min(random_job_index,other_job_index))
            solution.fix_objective()

        return solution

    @staticmethod
    def passive_swap(solution : SingleSolution, force_improve: bool = True):
        """Performs a swap between the 2 less effective jobs in terms of WSPT rule

        Args:
            solution (SingleSolution): Solution to be improved
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.

        Returns:
            SingleSolution: New solution
        """
        if(len(solution.machine.job_schedule)>1):

            jobs_list = list(range(solution.instance.n))
            jobs_list.sort(key = lambda job_id : float(solution.instance.W[job_id])/float(solution.instance.P[job_id]))
            job_i_id, job_j_id = jobs_list[0], jobs_list[1]
            machine_schedule = solution.machine.job_schedule
            machine_schedule_jobs_id = [job.id for job in machine_schedule]
            job_i_pos, job_j_pos = machine_schedule_jobs_id.index(job_i_id), machine_schedule_jobs_id.index(job_j_id)

            old_ci = solution.machine.objective

            new_ci = solution.machine.completion_time_swap(
                job_i_pos, job_j_pos, solution.instance)

            # Apply the move
            if not force_improve or (new_ci <= old_ci):
                machine_schedule[job_i_pos], machine_schedule[
                    job_j_pos] = machine_schedule[
                        job_j_pos], machine_schedule[job_i_pos]
                solution.machine.total_weighted_completion_time(solution.instance,min(job_i_pos,job_j_pos))
                solution.fix_objective()

        return solution
    
    @staticmethod
    def lahc_neighbour(solution : SingleSolution, objective : RootProblem.Objective):
        """Generates a neighbour solution of the given solution for the lahc metaheuristic

        Args:
            solution_i (SingleSolution): Solution to be improved

        Returns:
            SingleSolution: New solution
        """
        solution_copy = solution.copy()
        #for _ in range(1,random.randint(1, 2)):
        solution_copy = NeighbourhoodGeneration.random_swap(
            solution_copy, objective, force_improve=False)

        return solution_copy
