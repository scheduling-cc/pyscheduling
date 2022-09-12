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
class ParallelInstance(RootProblem.Instance):

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

    machine_num: int
    completion_time: int = 0
    last_job: int = -1
    job_schedule: list[Job] = field(default_factory=list)

    def __init__(self, machine_num: int, completion_time: int = 0, last_job: int = -1, job_schedule: list[Job] = None) -> None:
        """Constructor of Machine

        Args:
            machine_num (int): ID of the machine
            completion_time (int, optional): completion time of the last job of the machine. Defaults to 0.
            last_job (int, optional): ID of the last job set on the machine. Defaults to -1.
            job_schedule (list[Job], optional): list of Jobs scheduled on the machine in the exact given sequence. Defaults to None.
        """
        self.machine_num = machine_num
        self.completion_time = completion_time
        self.last_job = last_job
        if job_schedule is None:
            self.job_schedule = []
        else:
            self.job_schedule = job_schedule

    def __str__(self):
        return str(self.machine_num + 1) + " | " + " : ".join(map(str, [(job.id, job.start_time, job.end_time) for job in self.job_schedule])) + " | " + str(self.completion_time)

    def __eq__(self, other):
        same_machine = other.machine_num == self.machine_num
        same_schedule = other.job_schedule == self.job_schedule
        return (same_machine and same_schedule)

    def copy(self):
        return Machine(self.machine_num, self.completion_time, self.last_job, list(self.job_schedule))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    @staticmethod
    def fromDict(machine_dict):
        return Machine(machine_dict["machine_num"], machine_dict["completion_time"], machine_dict["last_job"], machine_dict["job_schedule"])

    def compute_completion_time(self, instance: ParallelInstance, startIndex: int = 0):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job and returns the final completion_time,
        works with both RmSijkCmax and RmriSijkCmax problems

        Args:
            instance (ParallelInstance): The instance associated to the machine
            startIndex (int) : The job index the function starts operating from

        Returns:
            int: completion_time of the machine
        """
        ci = 0
        if len(self.job_schedule) > 0:
            if startIndex > 0:
                prev_job, startTime, c_prev = self.job_schedule[startIndex - 1]
                job = self.job_schedule[startIndex].id
                if hasattr(instance, 'R'):
                    release_time = max(instance.R[job] - c_prev, 0)
                else:
                    release_time = 0
                startTime = c_prev + release_time
                ci = startTime + \
                    instance.S[self.machine_num][prev_job][job] + \
                    instance.P[job][self.machine_num]
            else:
                job = self.job_schedule[0].id

                if hasattr(instance, 'R'):
                    startTime = max(0, instance.R[job])
                else:
                    startTime = 0

                ci = startTime + instance.P[job][self.machine_num] + \
                    + instance.S[self.machine_num][job][job]  # Added Sk_ii for rabadi benchmark
            self.job_schedule[startIndex] = Job(job, startTime, ci)
            job_prev_i = job
            for i in range(startIndex+1, len(self.job_schedule)):
                job_i = self.job_schedule[i].id

                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job_i])
                else:
                    startTime = ci

                setup_time = instance.S[self.machine_num][job_prev_i][job_i]
                proc_time = instance.P[job_i][self.machine_num]
                ci = startTime + proc_time + setup_time

                self.job_schedule[i] = Job(job_i, startTime, ci)
                job_prev_i = job_i
        self.completion_time = ci
        return ci

    def completion_time_insert(self, job: int, pos: int, instance: ParallelInstance):
        """
        Computes the machine's completion time if we insert "job" at "pos" in the machine's job_schedule
        Args:
            job (int): id of the inserted job
            pos (int): position where the job is inserted in the machine
            instance (ParallelInstance): the current problem instance
        Returns:
            ci (int) : completion time
        """
        if pos > 0:  # There's at least one job in the schedule
            prev_job, startTime, c_prev = self.job_schedule[pos - 1]
            if hasattr(instance, 'R'):
                release_time = max(instance.R[job] - c_prev, 0)
            else:
                release_time = 0
            ci = c_prev + release_time + \
                instance.S[self.machine_num][prev_job][job] + \
                instance.P[job][self.machine_num]
        else:
            if hasattr(instance, 'R'):
                release_time = max(instance.R[job], 0)
            else:
                release_time = 0
            # First job to be inserted
            # Added Sk_ii for rabadi benchmark
            ci = release_time + \
                instance.P[job][self.machine_num] + \
                instance.S[self.machine_num][job][job]

        job_prev_i = job
        for i in range(pos, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            if hasattr(instance, 'R'):
                startTime = max(ci, instance.R[job_i])
            else:
                startTime = ci
            setup_time = instance.S[self.machine_num][job_prev_i][job_i]
            proc_time = instance.P[job_i][self.machine_num]
            ci = startTime + proc_time + setup_time

            job_prev_i = job_i

        return ci

    def completion_time_remove(self, pos: int, instance: ParallelInstance):
        """
        Computes the machine's completion time if we remove the job at "pos" in the machine's job_schedule
        Args:
            pos (int): position of the job to be removed
            instance (ParallelInstance): the current problem instance
        Returns:
            ci (int) : completion time
        """
        job_prev_i, ci = -1, 0
        if pos > 0:  # There's at least one job in the schedule
            job_prev_i, startTime, ci = self.job_schedule[pos - 1]

        for i in range(pos+1, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            if hasattr(instance, 'R'):
                startTime = max(ci, instance.R[job_i])
            else:
                startTime = ci
            setup_time = instance.S[self.machine_num][job_prev_i][job_i] if job_prev_i != -1 \
                else instance.S[self.machine_num][job_i][job_i]  # Added Sk_ii for rabadi
            proc_time = instance.P[job_i][self.machine_num]
            ci = startTime + proc_time + setup_time

            job_prev_i = job_i

        return ci

    def completion_time_remove_insert(self, pos_remove: int, job: int, pos_insert: int, instance:  ParallelInstance):
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

        job_prev_i, ci = -1, 0
        if first_pos > 0:  # There's at least one job in the schedule
            job_prev_i, startTime, ci = self.job_schedule[first_pos - 1]

        for i in range(first_pos, len(self.job_schedule)):
            job_i = self.job_schedule[i][0]

            # If job needs to be inserted to position i
            if i == pos_insert:
                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job])
                else:
                    startTime = ci
                setup_time = instance.S[self.machine_num][job_prev_i][job] if job_prev_i != -1 \
                    else instance.S[self.machine_num][job][job]
                proc_time = instance.P[job][self.machine_num]
                ci = startTime + proc_time + setup_time

                job_prev_i = job

            # If the job_i is not the one to be removed
            if i != pos_remove:
                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job_i])
                else:
                    startTime = ci
                setup_time = instance.S[self.machine_num][job_prev_i][job_i] if job_prev_i != -1 \
                    else instance.S[self.machine_num][job_i][job_i]
                proc_time = instance.P[job_i][self.machine_num]
                ci = startTime + proc_time + setup_time

                job_prev_i = job_i

        return ci

    def completion_time_swap(self, pos_i: int, pos_j: int, instance: ParallelInstance):
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

        job_prev_i, ci = -1, 0
        if first_pos > 0:  # There's at least one job in the schedule
            job_prev_i, startTime, ci = self.job_schedule[first_pos - 1]

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
            setup_time = instance.S[self.machine_num][job_prev_i][job_i] if job_prev_i != -1 \
                else instance.S[self.machine_num][job_i][job_i]
            proc_time = instance.P[job_i][self.machine_num]
            ci = startTime + proc_time + setup_time

            job_prev_i = job_i

        return ci


@dataclass
class ParallelSolution(RootProblem.Solution):

    machines: list[Machine]

    def __init__(self, instance: ParallelInstance = None, machines: list[Machine] = None, objective_value: int = 0):
        """Constructor of RmSijkCmax_Solution

        Args:
            instance (ParallelInstance, optional): Instance to be solved by the solution. Defaults to None.
            configuration (list[ParallelMachines.Machine], optional): list of machines of the instance. Defaults to None.
            objective_value (int, optional): initial objective value of the solution. Defaults to 0.
        """
        self.instance = instance
        if machines is None:
            self.machines = []
            for i in range(instance.m):
                machine = Machine(i, 0, -1, [])
                self.machines.append(machine)
        else:
            self.machines = machines
        self.objective_value = 0

    def __str__(self):
        return "Objective : " + str(self.objective_value) + "\n" + "Machine_ID | Job_schedule (job_id , start_time , completion_time) | Completion_time\n" + "\n".join(map(str, self.machines))

    def copy(self):
        copy_machines = []
        for m in self.machines:
            copy_machines.append(m.copy())

        copy_solution = ParallelSolution(self.instance)
        for i in range(self.instance.m):
            copy_solution.machines[i] = copy_machines[i]
        copy_solution.objective_value = self.objective_value
        return copy_solution

    def __lt__(self, other):
        if self.instance.get_objective().value > 0 :
            return self.objective_value < other.objective_value
        else : return other.objective_value < self.objective_value
    
    def cmax(self):
        """Sets the job_schedule of every machine associated to the solution and sets the objective_value of the solution to Cmax
            which equals to the maximal completion time of every machine
        """
        if self.instance != None:
            for k in range(self.instance.m):
                self.machines[k].compute_completion_time(self.instance)
        self.objective_value = max(
            [machine.completion_time for machine in self.machines])

    def tmp_cmax(self, temp_ci={}):
        """
        returns the cmax of a solution according to the the ci in the dict temp_ci if present, 
        if not it takes the ci of the machine, this doesn't modify the "cmax" of the machine.
        """
        this_cmax = 0
        for i in range(self.instance.m):
            ci = temp_ci.get(i, self.machines[i].completion_time)
            if ci > this_cmax:
                this_cmax = ci
        return this_cmax

    def fix_cmax(self):
        """Sets the objective_value of the solution to Cmax
            which equals to the maximal completion time of every machine
        """
        self.objective_value = max(
            [machine.completion_time for machine in self.machines])

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
        set_jobs = set()
        is_valid = True
        for machine in self.machines:
            prev_job = None
            ci, setup_time, expected_start_time = 0, 0, 0
            for i, element in enumerate(machine.job_schedule):
                job, startTime, endTime = element
                # Test End Time + start Time
                if hasattr(self.instance,'R'):
                    expected_start_time = max(self.instance.R[job],ci)
                else:
                    expected_start_time = ci
                if hasattr(self.instance,'S'):
                    if prev_job is None:
                        setup_time = self.instance.S[machine.machine_num][job][job]
                    else:
                        setup_time = self.instance.S[machine.machine_num][prev_job][job]
                else: setup_time = 0

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


class PM_LocalSearch(RootProblem.LocalSearch):
    @staticmethod
    def _inter_machine_insertion(solution: ParallelSolution):
        """For every job, verify if rescheduling it on the same machine at a different position or on a whole different machines gives a better solution

        Args:
            solution (ParallelSolution): The initial solution to be improved

        Returns:
            ParallelSolution: Improved solution
        """
        for i in range(solution.instance.m):  # For every machine in the system
            for l in range(solution.instance.m):  # For all the other machines
                move = None
                if (l != i) and len(solution.machines[i].job_schedule) > 1:
                    # Machine i
                    machine_i = solution.machines[i]
                    machine_i_schedule = machine_i.job_schedule
                    old_ci = machine_i.completion_time
                    # Machine L
                    machine_l = solution.machines[l]
                    machine_l_schedule = machine_l.job_schedule
                    old_cl = machine_l.completion_time
                    # for every job in the machine
                    for k in range(len(machine_i_schedule)):
                        job_k = machine_i_schedule[k]
                        ci = machine_i.completion_time_remove(k,solution.instance)
                        for j in range(len(machine_l_schedule)):
                            cl = machine_l.completion_time_insert(job_k.id,j,solution.instance)

                            cnd1 = (ci < old_ci and cl < old_cl)
                            cnd2 = (ci < old_ci and cl > old_cl and old_ci - ci >= cl - old_cl and cl !=
                                    solution.objective_value and cl <= solution.objective_value)
                            if cnd1:
                                if not move:
                                    move = (l, k, j, ci, cl)
                                elif ci-old_ci+cl-old_cl >= ci-move[3]+cl-move[4]:
                                    move = (l, k, j, ci, cl)
                            elif cnd2:
                                if not move:
                                    move = (l, k, j, ci, cl)
                                elif ci-old_ci+cl-old_cl >= ci-move[3]+cl-move[4]:
                                    move = (l, k, j, ci, cl)
                if move:
                    # Remove job k from machine i
                    job_k = machine_i_schedule.pop(move[1])
                    # Insert job k in machine l in pos j
                    machine_l_schedule.insert(move[2], job_k)
                    if machine_i.completion_time == solution.objective_value:
                        solution.objective_value = move[3]
                    # New completion time for machine i
                    machine_i.completion_time = move[3]
                    # New completion time for machine j
                    machine_l.completion_time = move[4]

                    # print(solution.cmax)
                    solution.cmax()
        solution.cmax()
        return solution

    @staticmethod
    def _internal_swap(solution: ParallelSolution):
        """Swap between 2 jobs on the same machine whose completion_time is maximal if it gives a better solution

        Args:
            solution (ParallelSolution): The initial solution to be improved

        Returns:
            ParallelSolution: Improved solution
        """
        cmax_machines_list = []
        for m, machine in enumerate(solution.machines):
            # if machine.completion_time == solution.cmax:
            cmax_machines_list.append(m)
        #print("Machines Cmax : " + str(len(cmax_machines_list)))
        for nb_machine in cmax_machines_list:
            cmax_machine = solution.machines[nb_machine]
            cmax_machine_schedule = cmax_machine.job_schedule
            move = None
            for i in range(0, len(cmax_machine_schedule)):
                for j in range(i+1, len(cmax_machine_schedule)):
                    new_ci = cmax_machine.completion_time_swap(i,j,solution.instance)
                    if new_ci < cmax_machine.completion_time:
                        if not move:
                            move = (i, j, new_ci)
                        elif new_ci < move[2]:
                            move = (i, j, new_ci)

            if move:
                # print(nb_machine,move)
                cmax_machine_schedule[move[0]], cmax_machine_schedule[move[1]
                                                                      ] = cmax_machine_schedule[move[1]], cmax_machine_schedule[move[0]]
                cmax_machine.completion_time = move[2]
                solution.cmax()
        solution.cmax()
        return solution

    @staticmethod
    def _external_swap(solution: ParallelSolution):
        """Swap between 2 jobs on different machines, where one of the machines has the maximal completion_time among all

        Args:
            solution (ParallelSolution): The initial solution to be improved

        Returns:
            ParallelSolution: Improved solution
        """
        cmax_machines_list = []
        other_machines = []
        for m, machine in enumerate(solution.machines):
            if machine.completion_time == solution.objective_value:
                cmax_machines_list.append(m)
            else:
                other_machines.append(m)
        for nb_machine in cmax_machines_list:
            cmax_machine = solution.machines[nb_machine]
            cmax_machine_schedule = cmax_machine.job_schedule
            old_ci = cmax_machine.completion_time

            move = None
            other_machines_copy = list(other_machines)

            while not move and len(other_machines_copy) != 0:

                random_index = random.randrange(len(other_machines_copy))
                other_machine_index = other_machines_copy.pop(random_index)
                other_machine = solution.machines[other_machine_index]
                other_nb_machine = other_machine.machine_num
                other_machine_schedule = other_machine.job_schedule

                old_cl = other_machine.completion_time
                old_cmax = solution.objective_value
                best_cmax = old_cmax
                best_diff = None

                for j in range(len(cmax_machine_schedule)):
                    for k in range(len(other_machine_schedule)):
                        job_cmax, _, _ = cmax_machine_schedule[j]
                        job_other_machine, _, _ = other_machine_schedule[k]

                        ci = cmax_machine.completion_time_remove_insert(j, job_other_machine, j, solution.instance)
                        cl = other_machine.completion_time_remove_insert(k, job_cmax, k, solution.instance)
                        
                        new_cmax = solution.tmp_cmax(temp_ci = {nb_machine: ci, other_nb_machine: cl})
                        #print(nb_machine, other_mahcine.machine_num, j, k, ci, cl, new_cmax, job_cmax, job_other_machine)
                        if new_cmax < old_cmax:
                            if not move:
                                move = (other_machine_index, j, k, ci, cl)
                                best_cmax = new_cmax
                            elif new_cmax < best_cmax:
                                move = (other_machine_index, j, k, ci, cl)
                                best_cmax = new_cmax
                        elif new_cmax == best_cmax and (ci < old_ci
                                                            or cl < old_cl):
                            if not move:
                                move = (other_machine_index, j, k, ci, cl)
                                best_diff = old_ci - ci + old_cl - cl
                            elif (not best_diff or old_ci - ci + old_cl - cl <
                                best_diff) and best_cmax == old_cmax:
                                move = (other_machine_index, j, k, ci, cl)
                                best_diff = old_ci - ci + old_cl - cl
            if move:  # Apply the best move
                cmax_machine_schedule[move[1]], other_machine_schedule[move[2]
                                                                       ] = other_machine_schedule[move[2]], cmax_machine_schedule[move[1]]
                cmax_machine.completion_time = move[3]
                other_machine.completion_time = move[4]
                solution.cmax()
                cmax_machine.compute_completion_time(solution.instance)
                other_machine.compute_completion_time(solution.instance)
                solution.fix_cmax()
        return solution

    @staticmethod
    def _external_insertion(solution: ParallelSolution):
        """Delete a job from the machine whose completion_time is maximal and insert it on another one

        Args:
            solution (ParallelSolution): The initial solution to be improved

        Returns:
            ParallelSolution: Improved solution
        """
        cmax_machines_list = []
        other_machines = []
        for m, machine in enumerate(solution.machines):
            if machine.completion_time == solution.objective_value:
                cmax_machines_list.append(m)
            else:
                other_machines.append(m)
        for nb_machine in cmax_machines_list:
            cmax_machine = solution.machines[nb_machine]
            cmax_machine_schedule = cmax_machine.job_schedule
            old_ci = cmax_machine.completion_time
            if len(cmax_machine_schedule) > 1:
                move = None
                other_machines_copy = list(other_machines)

                while not move and len(other_machines_copy) != 0:

                    random_index = random.randrange(len(other_machines_copy))
                    other_machine_index = other_machines_copy.pop(random_index)
                    other_machine = solution.machines[other_machine_index]
                    other_nb_machine = other_machine.machine_num
                    other_machine_schedule = other_machine.job_schedule

                    old_cl = other_machine.completion_time
                    old_cmax = solution.objective_value
                    best_cmax = old_cmax
                    best_diff = None

                    for j in range(len(cmax_machine_schedule)):
                        job_cmax, _, _ = cmax_machine_schedule[j]
                        ci = cmax_machine.completion_time_remove(j, solution.instance)
                        for k in range(len(other_machine_schedule)):
                            
                            cl = other_machine.completion_time_insert(job_cmax,k,solution.instance)

                            new_cmax = solution.tmp_cmax(temp_ci = {nb_machine: ci, other_nb_machine: cl})
                            if new_cmax < old_cmax:
                                if not move:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_cmax = new_cmax
                                    #print(1,old_cmax,old_ci,old_cl,move)
                                elif new_cmax < best_cmax:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_cmax = new_cmax
                                    #print(2,old_cmax,old_ci,old_cl,move)
                            elif new_cmax == best_cmax and (ci < old_ci
                                                                or cl < old_cl):
                                if not move:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_diff = old_ci - ci + old_cl - cl
                                    #print(3,old_cmax,old_ci,old_cl,move)
                                elif (not best_diff or old_ci - ci + old_cl - cl <
                                    best_diff) and best_cmax == old_cmax:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_diff = old_ci - ci + old_cl - cl
                                    #print(4,old_cmax,old_ci,old_cl,move)

                if move:  # Apply the best move
                    cmax_job = cmax_machine_schedule.pop(move[1])
                    other_machine_schedule.insert(move[2], cmax_job)
                    cmax_machine.compute_completion_time(solution.instance)
                    other_machine.compute_completion_time(solution.instance)
                    solution.fix_cmax()
        return solution

    @staticmethod
    def _balance(solution: ParallelSolution):
        """Reschedule jobs between machines in order to balance their completion_time thus giving a better solution

        Args:
            solution (ParallelSolution): The initial solution to be improved

        Returns:
            ParallelSolution: Improved solution
        """
        change = True
        while change:
            change = False
            cmax_machines_list = []
            other_machines = []
            for m, machine in enumerate(solution.machines):
                if machine.completion_time == solution.objective_value:
                    cmax_machines_list.append(machine)
                else:
                    other_machines.append(machine)

            for cmax_machine in cmax_machines_list:
                nb_machine = cmax_machine.machine_num
                cmax_machine_schedule = cmax_machine.job_schedule
                old_ci = cmax_machine.completion_time
                if len(cmax_machine_schedule) > 1:
                    other_machines = sorted(
                        other_machines,
                        key=lambda machine: machine.completion_time)

                    move = None
                    l = 0
                    while not move and l < len(other_machines):
                        other_machine_index = other_machines[l].machine_num
                        other_machine = solution.machines[other_machine_index]
                        other_nb_machine = other_machine.machine_num
                        other_mahcine_schedule = other_machine.job_schedule

                        old_cl = other_machine.completion_time
                        old_cmax = solution.objective_value
                        best_cmax = old_cmax
                        best_diff = None

                        j = len(cmax_machine_schedule) - 1
                        job_cmax, _, _ = cmax_machine_schedule[j]
                        ci = cmax_machine.completion_time_remove(j,solution.instance)
                        for k in range(len(other_mahcine_schedule)):
                            cl = other_machine.completion_time_insert(job_cmax, k, solution.instance)
                            
                            new_cmax = solution.tmp_cmax(temp_ci = {nb_machine: ci, other_nb_machine: cl})
                            if new_cmax < old_cmax:
                                if not move:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_cmax = new_cmax
                                    #print(1,old_cmax,old_ci,old_cl,move)
                                elif new_cmax < best_cmax:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_cmax = new_cmax
                                    #print(2,old_cmax,old_ci,old_cl,move)
                            elif new_cmax == best_cmax and (ci < old_ci
                                                                or cl < old_cl):
                                if not move:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_diff = old_ci - ci + old_cl - cl
                                    #print(3,old_cmax,old_ci,old_cl,move)
                                elif (not best_diff or old_ci - ci + old_cl - cl <
                                    best_diff) and best_cmax == old_cmax:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_diff = old_ci - ci + old_cl - cl
                                    #print(4,old_cmax,old_ci,old_cl,move)
                        l += 1
                    if move:  # Apply the best move
                        change = True
                        cmax_job = cmax_machine_schedule.pop(move[1])
                        other_mahcine_schedule.insert(move[2], cmax_job)
                        cmax_machine.compute_completion_time(solution.instance)
                        other_machine.compute_completion_time(solution.instance)
                        solution.fix_cmax()
        return solution

    @staticmethod
    def best_insertion_machine(solution : ParallelSolution,machine_id : int, job_id : int):
        """Find the best position to insert a job job_id in the machine machine_id

        Args:
            solution (ParallelSolution): Solution to be improved
            machine_id (int): ID of the machine 
            job_id (int): ID of the job

        Returns:
            ParallelSolution: New solution
        """
        machine = solution.machines[machine_id]
        machine_schedule = machine.job_schedule
        best_cl = None
        taken_move = 0
        for j in range(len(machine_schedule)):  # for every position in other machine
            cl = machine.completion_time_insert(job_id, j, solution.instance)

            if not best_cl or cl < best_cl:
                best_cl = cl
                taken_move = j

        machine_schedule.insert(taken_move, Job(job_id, 0, 0))
        machine.completion_time = machine.compute_completion_time(solution.instance)

        return solution


class NeighbourhoodGeneration():

    @staticmethod
    def random_swap(solution: ParallelSolution, force_improve: bool = True, internal: bool = False):
        """Performs a random swap between 2 jobs on the same machine or on different machines

        Args:
            solution (ParallelSolution): Solution to be improved
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.
            internal (bool, optional): If true, applies the swap between jobs on the same machine only. Defaults to False.

        Returns:
            ParallelSolution: New solution
        """
        # Get compatible machines (len(job_schedule) >= 1)
        compatible_machines = []
        for m in range(solution.instance.m):
            if (len(solution.machines[m].job_schedule) >= 1 and not internal) or \
                    (len(solution.machines[m].job_schedule) >= 2 and internal):
                compatible_machines.append(m)

        if len(compatible_machines) >= 2:

            random_machine_index = random.choice(compatible_machines)
            if internal:
                other_machine_index = random_machine_index
            else:
                other_machine_index = random.choice(compatible_machines)
                while other_machine_index == random_machine_index:
                    other_machine_index = random.choice(compatible_machines)

            random_machine = solution.machines[random_machine_index]
            other_machine = solution.machines[other_machine_index]

            random_machine_schedule = random_machine.job_schedule
            other_machine_schedule = other_machine.job_schedule

            old_ci, old_cl = random_machine.completion_time, other_machine.completion_time

            random_job_index = random.randrange(len(random_machine_schedule))
            other_job_index = random.randrange(len(other_machine_schedule))

            if internal:  # Internal swap
                while other_job_index == random_job_index:
                    other_job_index = random.randrange(
                        len(other_machine_schedule))

                new_ci = random_machine.completion_time_swap(
                    random_job_index, other_job_index, solution.instance)
                new_cl = new_ci
            else:  # External swap

                job_random, _, _ = random_machine_schedule[random_job_index]
                other_job, _, _ = other_machine_schedule[other_job_index]

                new_ci = random_machine.completion_time_remove_insert(
                    random_job_index, other_job, random_job_index, solution.instance)
                new_cl = other_machine.completion_time_remove_insert(
                    other_job_index, job_random, other_job_index, solution.instance)

            # Apply the move
            if not force_improve or (new_ci + new_cl <= old_ci + old_cl):
                random_machine_schedule[random_job_index], other_machine_schedule[
                    other_job_index] = other_machine_schedule[
                        other_job_index], random_machine_schedule[random_job_index]
                random_machine.completion_time = random_machine.compute_completion_time(
                    solution.instance)
                other_machine.completion_time = other_machine.compute_completion_time(
                    solution.instance)
                solution.fix_cmax()

        return solution

    @staticmethod
    def random_inter_machine_insertion(solution: ParallelSolution, force_improve: bool = True):
        """Removes randomly a job from a machine and insert it on the same machine in different possition or another machine

        Args:
            solution (ParallelSolution): Solution to be improved
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.

        Returns:
            ParallelSolution: New solution
        """
        # Get compatible machines (len(job_schedule) >= 2)
        compatible_machines = []
        for m in range(solution.instance.m):
            if (len(solution.machines[m].job_schedule) >= 2):
                compatible_machines.append(m)

        if len(compatible_machines) >= 1:

            random_machine_index = random.choice(compatible_machines)
            other_mahcine_index = random.randrange(solution.instance.m)
            while other_mahcine_index == random_machine_index:
                other_mahcine_index = random.randrange(solution.instance.m)

            random_machine = solution.machines[random_machine_index]
            other_machine = solution.machines[other_mahcine_index]

            random_machine_schedule = random_machine.job_schedule
            other_machine_schedule = other_machine.job_schedule

            random_job_index = random.randrange(len(random_machine_schedule))
            other_job_index = random.randrange(len(other_machine_schedule)) if len(
                other_machine_schedule) > 0 else 0

            old_ci, old_cl = random_machine.completion_time, other_machine.completion_time
            job_i, _, _ = random_machine_schedule[random_job_index]

            new_ci = random_machine.completion_time_remove(
                random_job_index, solution.instance)
            new_cl = other_machine.completion_time_insert(
                job_i, other_job_index, solution.instance)

            # Apply the move
            if not force_improve or (new_ci + new_cl <= old_ci + old_cl):
                job_i = random_machine_schedule.pop(random_job_index)
                other_machine_schedule.insert(other_job_index, job_i)

                random_machine.completion_time = random_machine.compute_completion_time(
                    solution.instance)
                other_machine.completion_time = other_machine.compute_completion_time(
                    solution.instance)

                solution.fix_cmax()

        return solution

    @staticmethod
    def restricted_swap(solution: ParallelSolution):
        """Performs a random swap between 2 jobs of 2 different machines whose completion time is equal
        to the maximal completion time. If it's not possible, performs the move between a job on
        the machine whose completion time is equel to the maximal completion time and another
        one

        Args:
            solution (ParallelSolution): Solution to be improved

        Returns:
            ParallelSolution: New solution
        """
        cmax_machines_list = []
        other_machines = []
        for m, machine in enumerate(solution.machines):
            if machine.completion_time == solution.objective_value:
                cmax_machines_list.append(m)
            elif len(machine.job_schedule
                     ) >= 1:  # Compatible machines have len > 1:
                other_machines.append(m)

        if len(cmax_machines_list) > 2:
            choices = random.sample(cmax_machines_list, 2)
            m1, m2 = choices[0], choices[1]
        elif len(cmax_machines_list) == 2:
            m1, m2 = cmax_machines_list[0], cmax_machines_list[1]
        else:
            m1 = cmax_machines_list[0]
            if len(other_machines) > 0:
                m2 = random.choice(other_machines)
            else:
                return solution

        t1 = random.randrange(len(solution.machines[m1].job_schedule))
        t2 = random.randrange(len(solution.machines[m2].job_schedule))

        machine_1_schedule = solution.machines[m1].job_schedule
        machine_2_schedule = solution.machines[m2].job_schedule

        machine_1_schedule[t1], machine_2_schedule[t2] = machine_2_schedule[
            t2], machine_1_schedule[t1]

        solution.machines[m1].completion_time = solution.machines[m1].compute_completion_time(
            solution.instance)
        solution.machines[m2].completion_time = solution.machines[m2].compute_completion_time(
            solution.instance)

        solution.fix_cmax()
        return solution

    @staticmethod
    def restricted_insert(solution: ParallelSolution):
        """Performs a random inter_machine_insertion between 2 different machines whose
        completion time is equal to the maximal completion time. If it's not possible, performs the
        move between a job on the machine whose completion time is equel to the 
        maximal completion time and another one

        Args:
            solution (ParallelSolution): Solution to be improved

        Returns:
            ParallelSolution: New solution
        """
        cmax_machines_list = []
        other_machines = []
        for m, machine in enumerate(solution.machines):
            if machine.completion_time == solution.objective_value:
                cmax_machines_list.append(m)
            else:
                other_machines.append(m)

        if len(cmax_machines_list) > 2:
            choices = random.sample(cmax_machines_list, 2)
            m1, m2 = choices[0], choices[1]
        elif len(cmax_machines_list) == 2:
            m1, m2 = cmax_machines_list[0], cmax_machines_list[1]
        else:
            m1 = cmax_machines_list[0]
            m2 = random.choice(other_machines)

        t1 = random.randrange(len(solution.machines[m1].job_schedule))
        t2 = random.randrange(len(solution.machines[m2].job_schedule)) if len(
            solution.machines[m2].job_schedule) > 0 else 0

        machine_1_schedule = solution.machines[m1].job_schedule
        machine_2_schedule = solution.machines[m2].job_schedule

        job_i = machine_1_schedule.pop(t1)
        machine_2_schedule.insert(t2, job_i)

        solution.machines[m1].completion_time = solution.machines[m1].compute_completion_time(
            solution.instance)
        solution.machines[m2].completion_time = solution.machines[m2].compute_completion_time(
            solution.instance)

        solution.fix_cmax()
        return solution

    @staticmethod
    def lahc_neighbour(solution_i):
        """Generates a neighbour solution of the given solution for the lahc metaheuristic

        Args:
            solution_i (ParallelSolution): Solution to be improved

        Returns:
            ParallelSolution: New solution
        """
        solution = solution_i.copy()

        r = random.random()
        if r < 0.5:
            solution = NeighbourhoodGeneration.random_swap(
                solution, force_improve=False, internal=False)
        else:
            solution = NeighbourhoodGeneration.random_swap(
                solution, force_improve=False, internal=True)
        return solution

    @staticmethod
    def SA_neighbour(solution: ParallelSolution):
        """Generates a neighbour solution of the given solution for the SA metaheuristic

        Args:
            solution_i (ParallelSolution): Solution to be improved

        Returns:
            ParallelSolution: New solution
        """
        solution_copy = solution.copy()
        r = random.random()
        if r < 0.33:
            solution_copy = NeighbourhoodGeneration.random_swap(
                solution_copy, force_improve=False, internal=False)  # External Swap
        elif r < 0.67:
            solution_copy = NeighbourhoodGeneration.random_swap(solution_copy, force_improve=False,
                                                                internal=True)  # Internal Swap
        else:
            solution_copy = NeighbourhoodGeneration.random_inter_machine_insertion(
                solution_copy, force_improve=False)  # Inter Machine Insertion
        return solution_copy

    @staticmethod
    def RSA_neighbour(solution: ParallelInstance, q0: float):
        """Generates a neighbour solution of the given solution for the lahc metaheuristic

        Args:
            solution_i (ParallelSolution): Solution to be improved
            q0 (float): Probability to apply restricted swap compared to
            restricted insertion.

        Returns:
            ParallelSolution: New solution
        """
        solution_copy = solution.copy()
        r = random.random()
        if r < q0:
            solution_copy = NeighbourhoodGeneration.restricted_swap(
                solution_copy)
        r = random.random()
        if r < q0:
            solution_copy = NeighbourhoodGeneration.restricted_insert(
                solution_copy)
        return solution_copy