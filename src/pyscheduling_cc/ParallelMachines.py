import json
import random
from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

import pyscheduling_cc.Problem as Problem

Job = namedtuple('Job', ['id', 'start_time', 'completion_time'])


class GenerationProtocol(Enum):
    VALLADA = 1


class GenerationLaw(Enum):
    UNIFORM = 1
    NORMAL = 2


@dataclass
class ParallelInstance(Problem.Instance):

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
            P_k = [int(ligne[j]) for j in range(1, 2*self.m, 2)]
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
        ligne = content[i].split('\t')
        ri = []  # Table : Release time of job i
        for j in range(2, len(ligne), 2):
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
        while i != len(content):
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
        ligne = content[i].split('\t')
        di = []  # Table : Due time of job i
        for j in range(2, len(ligne), 2):
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
        return str(self.machine_num + 1) + " | " + " : ".join(map(str, [(job.id, job.start_time, job.completion_time) for job in self.job_schedule])) + " | " + str(self.completion_time)

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

    @abstractmethod
    def compute_completion_time(self, instance: ParallelInstance):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job and returns the final completion_time,
        works with both RmSijkCmax and RmriSijkCmax problems

        Args:
            instance (ParallelInstance): The instance associated to the machine

        Returns:
            int: completion_time of the machine
        """
        ci = 0
        if len(self.job_schedule) > 0:
            first_job = self.job_schedule[0][0]
            if hasattr(instance, 'R'):
                startTime = max(0, instance.R[first_job])
            else:
                startTime = 0
            ci = startTime + instance.P[first_job][self.machine_num] + \
                + instance.S[self.machine_num][first_job][first_job]  # Added Sk_ii for rabadi benchmark
            self.job_schedule[0] = Job(first_job, startTime, ci)
            job_prev_i = first_job
            for i in range(1, len(self.job_schedule)):
                job_i = self.job_schedule[i][0]

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


@dataclass
class ParallelSolution(Problem.Solution):

    machines_number: int
    configuration: list[Machine]

    def __init__(self, m: int, instance: ParallelInstance = None):
        """Constructor of ParallelSolution

        Args:
            m (int): number of machines
            instance (ParallelInstance, optional): Instance to be solved by the solution. Defaults to None.
        """
        self.machines_number = m
        self.configuration = []
        for i in range(m):
            machine = Machine(i, 0, -1, [])
            self.configuration.append(machine)
        self.objective_value = 0

    def __str__(self):
        return "Objective : " + str(self.objective_value) + "\n" + "Machine_ID | Job_schedule (job_id , start_time , completion_time) | Completion_time\n" + "\n".join(map(str, self.configuration))

    def copy(self):
        copy_machines = []
        for m in self.configuration:
            copy_machines.append(m.copy())

        copy_solution = ParallelSolution(self.m)
        for i in range(self.m):
            copy_solution.configuration[i] = copy_machines[i]
        copy_solution.objective_value = self.objective_value
        return copy_solution

    def cmax(self):
        """Sets the job_schedule of every machine associated to the solution and sets the objective_value of the solution to Cmax
            which equals to the maximal completion time of every machine
        """
        if self.instance != None:
            for k in range(self.instance.m):
                self.configuration[k].compute_completion_time(self.instance)
        self.objective_value = max(
            [machine.completion_time for machine in self.configuration])

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


class PM_LocalSearch(Problem.LocalSearch):
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
                if (l != i) and len(solution.configuration[i].job_schedule) > 1:
                    # Machine i
                    machine_i = solution.configuration[i]
                    machine_i_schedule = machine_i.job_schedule
                    old_ci = machine_i.completion_time
                    # Machine L
                    machine_l = solution.configuration[l]
                    machine_l_schedule = machine_l.job_schedule
                    old_cl = machine_l.completion_time
                    # for every job in the machine
                    for k in range(len(machine_i_schedule)):
                        job_schedule_copy = list(machine_i_schedule)
                        new_schedule = Machine(
                            i, machine_i.completion_time, machine_i.last_job, job_schedule_copy)
                        job_k = new_schedule.job_schedule.pop(k)
                        ci = machine_i.compute_completion_time(
                            solution.instance)
                        for j in range(len(machine_l_schedule)):
                            job_schedule_copy_l = list(machine_l_schedule)
                            new_schedule_l = Machine(
                                l, machine_l.completion_time, machine_l.last_job, job_schedule_copy_l)
                            new_schedule_l.job_schedule.insert(j, job_k)
                            cl = machine_l.compute_completion_time(
                                solution.instance)

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
        for m, machine in enumerate(solution.configuration):
            # if machine.completion_time == solution.cmax:
            cmax_machines_list.append(m)
        #print("Machines Cmax : " + str(len(cmax_machines_list)))
        for nb_machine in cmax_machines_list:
            cmax_machine = solution.configuration[nb_machine]
            cmax_machine_schedule = cmax_machine.job_schedule
            move = None
            for i in range(0, len(cmax_machine_schedule)):
                for j in range(i+1, len(cmax_machine_schedule)):
                    job_schedule_copy = list(cmax_machine_schedule)
                    new_schedule = Machine(
                        nb_machine, cmax_machine.completion_time, cmax_machine.last_job, job_schedule_copy)
                    new_schedule.job_schedule[i], new_schedule.job_schedule[
                        j] = new_schedule.job_schedule[j], new_schedule.job_schedule[i]
                    new_ci = new_schedule.compute_completion_time(
                        solution.instance)
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
        for m, machine in enumerate(solution.configuration):
            if machine.completion_time == solution.objective_value:
                cmax_machines_list.append(m)
            else:
                other_machines.append(m)
        for nb_machine in cmax_machines_list:
            cmax_machine = solution.configuration[nb_machine]
            cmax_machine_schedule = cmax_machine.job_schedule
            old_ci = cmax_machine.completion_time

            move = None
            other_machines_copy = list(other_machines)

            while not move and len(other_machines_copy) != 0:

                random_index = random.randrange(len(other_machines_copy))
                other_machine_index = other_machines_copy.pop(random_index)
                other_machine = solution.configuration[other_machine_index]
                other_machine_schedule = other_machine.job_schedule

                old_cl = other_machine.completion_time
                old_cmax = solution.objective_value
                best_cmax = old_cmax
                best_diff = None

                for j in range(len(cmax_machine_schedule)):
                    for k in range(len(other_machine_schedule)):
                        new_machine_cmax = Machine(
                            nb_machine, cmax_machine.completion_time, cmax_machine.last_job, list(cmax_machine_schedule))
                        job_cmax = new_machine_cmax.job_schedule.pop(j)

                        new_other_machine = Machine(
                            other_machine_index, other_machine.completion_time, other_machine.last_job, list(other_machine_schedule))
                        job_other_machine = new_other_machine.job_schedule.pop(
                            k)

                        new_machine_cmax.job_schedule.insert(
                            j, job_other_machine)
                        new_other_machine.job_schedule.insert(k, job_cmax)

                        ci = new_machine_cmax.compute_completion_time(
                            solution.instance)
                        cl = new_other_machine.compute_completion_time(
                            solution.instance)
                        new_machine_cmax.completion_time = ci
                        new_other_machine.completion_time = cl

                        solution.configuration[nb_machine] = new_machine_cmax
                        solution.configuration[other_machine_index] = new_other_machine

                        solution.cmax()
                        if solution.objective_value < old_cmax:
                            if not move:
                                move = (other_machine_index, j, k, ci, cl)
                                best_cmax = solution.objective_value
                            elif solution.objective_value < best_cmax:
                                move = (other_machine_index, j, k, ci, cl)
                                best_cmax = solution.objective_value
                        elif solution.objective_value == best_cmax and (ci < old_ci or cl < old_cl):
                            if not move:
                                move = (other_machine_index, j, k, ci, cl)
                                best_diff = old_ci - ci + old_cl - cl
                            elif (not best_diff or old_ci - ci + old_cl - cl < best_diff) and best_cmax == old_cmax:
                                move = (other_machine_index, j, k, ci, cl)
                                best_diff = old_ci - ci + old_cl - cl
                        solution.configuration[nb_machine] = cmax_machine
                        solution.configuration[other_machine_index] = other_machine
                        solution.cmax()
            if move:  # Apply the best move
                cmax_machine_schedule[move[1]], other_machine_schedule[move[2]
                                                                       ] = other_machine_schedule[move[2]], cmax_machine_schedule[move[1]]
                cmax_machine.completion_time = move[3]
                other_machine.completion_time = move[4]
                solution.cmax()
        solution.cmax()
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
        for m, machine in enumerate(solution.configuration):
            if machine.completion_time == solution.objective_value:
                cmax_machines_list.append(m)
            else:
                other_machines.append(m)
        for nb_machine in cmax_machines_list:
            cmax_machine = solution.configuration[nb_machine]
            cmax_machine_schedule = cmax_machine.job_schedule
            old_ci = cmax_machine.completion_time
            if len(cmax_machine_schedule) > 1:
                move = None
                other_machines_copy = list(other_machines)

                while not move and len(other_machines_copy) != 0:

                    random_index = random.randrange(len(other_machines_copy))
                    other_machine_index = other_machines_copy.pop(random_index)
                    other_machine = solution.configuration[other_machine_index]
                    other_machine_schedule = other_machine.job_schedule

                    old_cl = other_machine.completion_time
                    old_cmax = solution.objective_value
                    best_cmax = old_cmax
                    best_diff = None

                    for j in range(len(cmax_machine_schedule)):
                        for k in range(len(other_machine_schedule)):
                            new_machine_cmax = Machine(
                                nb_machine, cmax_machine.completion_time, cmax_machine.last_job, list(cmax_machine_schedule))
                            job_cmax = new_machine_cmax.job_schedule.pop(j)

                            new_other_machine = Machine(
                                other_machine_index, other_machine.completion_time, other_machine.last_job, list(other_machine_schedule))
                            new_other_machine.job_schedule.insert(k, job_cmax)

                            ci = new_machine_cmax.compute_completion_time(
                                solution.instance)
                            cl = new_other_machine.compute_completion_time(
                                solution.instance)
                            new_machine_cmax.completion_time = ci
                            new_other_machine.completion_time = cl

                            solution.configuration[nb_machine] = new_machine_cmax
                            solution.configuration[other_machine_index] = new_other_machine

                            solution.cmax()
                            if solution.objective_value < old_cmax:
                                if not move:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_cmax = solution.objective_value
                                    # print(1,old_cmax,old_ci,old_cl,move)
                                elif solution.objective_value < best_cmax:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_cmax = solution.objective_value
                                    # print(2,old_cmax,old_ci,old_cl,move)
                            elif solution.objective_value == best_cmax and (ci < old_ci or cl < old_cl):
                                if not move:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_diff = old_ci - ci + old_cl - cl
                                    # print(3,old_cmax,old_ci,old_cl,move)
                                elif (not best_diff or old_ci - ci + old_cl - cl < best_diff) and best_cmax == old_cmax:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_diff = old_ci - ci + old_cl - cl
                                    # print(4,old_cmax,old_ci,old_cl,move)
                            solution.configuration[nb_machine] = cmax_machine
                            solution.configuration[other_machine_index] = other_machine
                            solution.cmax()
                if move:  # Apply the best move
                    cmax_job = cmax_machine_schedule.pop(move[1])
                    other_machine_schedule.insert(move[2], cmax_job)
                    cmax_machine.completion_time = move[3]
                    other_machine.completion_time = move[4]
                    solution.cmax()
        solution.cmax()
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
            for m, machine in enumerate(solution.configuration):
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
                        other_machines, key=lambda machine: machine.completion_time)

                    move = None
                    l = 0
                    while not move and l < len(other_machines):
                        other_machine_index = other_machines[l].machine_num
                        other_machine = solution.configuration[other_machine_index]
                        other_machine_schedule = other_machine.job_schedule

                        old_cl = other_machine.completion_time
                        old_cmax = solution.objective_value
                        best_cmax = old_cmax
                        best_diff = None

                        j = len(cmax_machine_schedule)-1
                        for k in range(len(other_machine_schedule)):
                            new_machine_cmax = Machine(
                                nb_machine, cmax_machine.completion_time, cmax_machine.last_job, list(cmax_machine_schedule))
                            job_cmax = new_machine_cmax.job_schedule.pop(j)

                            new_other_machine = Machine(
                                other_machine_index, other_machine.completion_time, other_machine.last_job, list(other_machine_schedule))
                            new_other_machine.job_schedule.insert(k, job_cmax)

                            ci = new_machine_cmax.compute_completion_time(
                                solution.instance)
                            cl = new_other_machine.compute_completion_time(
                                solution.instance)
                            new_machine_cmax.completion_time = ci
                            new_other_machine.completion_time = cl

                            solution.configuration[nb_machine] = new_machine_cmax
                            solution.configuration[other_machine_index] = new_other_machine

                            solution.cmax()
                            if solution.objective_value < old_cmax:
                                if not move:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_cmax = solution.objective_value
                                    # print(1,old_cmax,old_ci,old_cl,move)
                                elif solution.objective_value < best_cmax:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_cmax = solution.objective_value
                                    # print(2,old_cmax,old_ci,old_cl,move)
                            elif solution.objective_value == best_cmax and (ci < old_ci or cl < old_cl):
                                if not move:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_diff = old_ci - ci + old_cl - cl
                                    # print(3,old_cmax,old_ci,old_cl,move)
                                elif (not best_diff or old_ci - ci + old_cl - cl < best_diff) and best_cmax == old_cmax:
                                    move = (other_machine_index, j, k, ci, cl)
                                    best_diff = old_ci - ci + old_cl - cl
                                    # print(4,old_cmax,old_ci,old_cl,move)
                            solution.configuration[nb_machine] = cmax_machine
                            solution.configuration[other_machine_index] = other_machine
                            solution.cmax()
                        l += 1
                    if move:  # Apply the best move
                        change = True
                        cmax_job = cmax_machine_schedule.pop(move[1])
                        other_machine_schedule.insert(move[2], cmax_job)
                        cmax_machine.completion_time = move[3]
                        other_machine.completion_time = move[4]
                        solution.cmax()
            solution.cmax()
        return solution
