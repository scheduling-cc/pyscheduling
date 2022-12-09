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

    def generate_D(self, protocol: GenerationProtocol, law: GenerationLaw, PJobs: list[float], Pmin: int, Pmax: int, due_time_factor: float):
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
            if hasattr(self, 'R'):
                startTime = self.R[j] + PJobs[j]
            else:
                startTime = PJobs[j]
            if law.name == "UNIFORM":  # Generate uniformly
                n = int(random.uniform(
                    startTime, startTime + due_time_factor * sumP))

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
    
    objective_value: int = 0
    last_job: int = -1
    oper_schedule: list[Job] = field(default_factory=list)
    
    def __init__(self, machine_num:int, oper_schedule: list[Job] = None, last_job: int = -1, objective_value: int = 0) -> None:
        """Constructor of Machine

        Args:
            objective (int, optional): completion time of the last job of the machine. Defaults to 0.
            last_job (int, optional): ID of the last job set on the machine. Defaults to -1.
            job_schedule (list[Job], optional): list of Jobs scheduled on the machine in the exact given sequence. Defaults to None.
        """
        self.machine_num = machine_num
        self.objective_value = objective_value
        self.last_job = last_job
        if oper_schedule is None:
            self.oper_schedule = []
        else:
            self.oper_schedule = oper_schedule

    def __str__(self):
        return " : ".join(map(str, [(job.id, job.start_time, job.end_time) for job in self.oper_schedule])) + " | " + str(self.objective_value)

    def __eq__(self, other):
        same_schedule = other.oper_schedule == self.oper_schedule
        return (same_schedule)

    def copy(self):
        return Machine(self.machine_num, list(self.oper_schedule), self.last_job, self.objective_value)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    @staticmethod
    def fromDict(machine_dict):
        return Machine(machine_dict["objective"], machine_dict["last_job"], machine_dict["job_schedule"])

    def compute_current_ci(self, instance: FlowShopInstance, prev_machine_ci: int ,prev_job_ci: int, job_prev_i: int, job_i: int):
        """Computes the current ci when job_i comes after job_prev_i.
        This takes into account if we have setup times and release dates.

        Args:
            instance (SingelInstance): the instance to be solved.
            prev_machine_ci (int): the ci of the same job on the previous machine
            prev_ci (int): the completion time of the previous job on the same machine
            job_prev_i (int): id of the job that precedes the inserted job 
            job_i (int): id of the job to be inserted at the end

        Returns:
            tuple: (ci, start_time), the new completion time and start_time of the inserted job.
        """
        startTime = max(prev_machine_ci, prev_job_ci, instance.R[job_i]) if hasattr(instance, 'R') else max(prev_machine_ci, prev_job_ci)
        setupTime = instance.S[self.machine_num][job_prev_i][job_i] if hasattr(instance, 'S') else 0
        remaining_setupTime = max(setupTime-(startTime-prev_job_ci),0)
        proc_time = instance.P[job_i][self.machine_num]

        ci = startTime + remaining_setupTime + proc_time
        return startTime, ci
    
    def fix_schedule(self, instance: FlowShopInstance, prev_machine, startIndex: int = 0):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job

        Args:
            instance (SingleInstance): The instance associated to the machine
            startIndex (int) : The job index the function starts operating from

        Returns:
            int: objective
        """
        if startIndex > 0:
            job_prev_i, ci = self.oper_schedule[startIndex - 1].id, self.oper_schedule[startIndex - 1].end_time
        else:
            job_prev_i, ci = (self.oper_schedule[startIndex].id, 0)
        
        for i in range(startIndex, len(self.oper_schedule)):
            job_i = self.oper_schedule[i].id
            prev_machine_ci = prev_machine.oper_schedule[i].end_time if prev_machine is not None else 0
            start_time, ci = self.compute_current_ci(instance, prev_machine_ci, ci, job_prev_i, job_i)
            self.oper_schedule[i] = Job(job_i, start_time, ci)
            job_prev_i = job_i

        self.last_job = job_i
        self.objective_value = ci
        return ci

    def idle_time(self):
        """returns the idle time on the machine

        Returns:
            int: idle time of the machine
        """
        idleTime = self.oper_schedule[0].start_time
        for job_index in range(len(self.oper_schedule)-1):
            idleTime += self.oper_schedule[job_index+1].start_time - self.oper_schedule[job_index].end_time
        return idleTime

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
                machine = Machine(i, [], -1, 0)
                self.machines.append(machine)
        else:
            self.machines = machines
        if job_schedule is None: 
            self.job_schedule = []
        else: 
            self.job_schedule = job_schedule
        self.objective_value = 0

    def __str__(self):
        return "Objective : " + str(self.objective_value) + "\n" + "Jobs sequence : " + "\t".join(map(str, self.job_schedule)) + "\n" + "Machine_ID | Job_schedule (job_id , start_time , completion_time) | Completion_time\n" + "\n".join(map(str, self.machines))

    def copy(self):
        copy_machines = []
        for m in self.machines:
            copy_machines.append(m.copy())

        copy_solution = FlowShopSolution(self.instance, copy_machines,
                                        list(self.job_schedule), self.objective_value)
        return copy_solution

    def __lt__(self, other):
        if self.instance.get_objective().value > 0 :
            return self.objective_value < other.objective_value
        else : return other.objective_value < self.objective_value
    
    def propagate_schedule(self, startIndex: int=0):
        """Fills the job_schedule of every machine from job_schedule of Solution
        """
        if startIndex == 0: 
            for machine in self.machines :
                machine.oper_schedule = [Job(job.id,0,0) for job in self.job_schedule]

        elif len(self.job_schedule) != len(self.machines[0].oper_schedule): # Inserting one job only
            inserted_job_id = self.job_schedule[startIndex].id
            for machine in self.machines :
                machine.oper_schedule.insert(startIndex,Job(inserted_job_id,0,0))

        else: # Change starting from startIndex
            for machine in self.machines:
                for i in range(startIndex, len(self.job_schedule)):
                    job_id, start, end = self.job_schedule[i]
                    machine.oper_schedule[i] = Job(job_id, start, end)

    def fix_objective(self):
        objective = self.instance.get_objective()
        if objective == RootProblem.Objective.Cmax:
            self.objective_value = self.machines[-1].objective_value
        elif objective == RootProblem.Objective.wiCi:
            self.objective_value = sum( self.instance.W[job.id] * job.end_time for job in self.machines[-1].oper_schedule )
        elif objective == RootProblem.Objective.wiFi:
            self.objective_value = sum( self.instance.W[job.id] * (job.end_time - self.instance.R[job.id]) for job in self.machines[-1].oper_schedule )
        elif objective == RootProblem.Objective.wiTi:
            self.objective_value = sum( self.instance.W[job.id] * max(job.end_time-self.instance.D[job.id],0) for job in self.machines[-1].oper_schedule )
        elif objective == RootProblem.Objective.Lmax:
            self.objective_value = max( 0, max( job.end_time-self.instance.D[job.id] for job in self.machines[-1].oper_schedule ) )

        return self.objective_value

    def compute_objective(self, startIndex: int = 0):
        self.propagate_schedule(startIndex)
        prev_machine = None
        for i, machine in enumerate(self.machines):
            machine.fix_schedule(self.instance, prev_machine, startIndex)
            prev_machine = machine
        
        for j in range(len(self.job_schedule)):
            job_id = self.job_schedule[j].id
            start_time = self.machines[0].oper_schedule[j].start_time
            end_time = self.machines[-1].oper_schedule[j].end_time
            self.job_schedule[j] = Job( job_id, start_time, end_time )

        return self.fix_objective()

    def simulate_insert_last(self, job_id : int):
        """returns start_time and completion_time of job_id if scheduled at the end of job_schedule
        at every stage (machine)

        Args:
            job_id (int): job to be scheduled at the end

        Returns:
            int, int: start_time of job_id, completion_time of job_id
        """ 
        prev_machine_ci = 0
        for i, machine in enumerate(self.machines):
            prev_job, _, prev_ci = machine.oper_schedule[-1] if len(machine.oper_schedule) > 0 else (job_id, 0, 0)
            start_time, end_time = machine.compute_current_ci(self.instance, prev_machine_ci, prev_ci, prev_job, job_id)
            prev_machine_ci = end_time

        return start_time, end_time
    
    def simulate_insert_objective(self, job_id, start_time, end_time):
        """Returns the new objective if job_id is inserted at the end with start_time and end_time

        Args:
            job_id (int): id of the inserted job
            start_time (int): start time of the job
            end_time (int): end time of the job
        
        Returns:
            int: the new objective
        """
        objective = self.instance.get_objective()
        if objective == RootProblem.Objective.Cmax:
            return end_time
        elif objective == RootProblem.Objective.wiCi:
            return self.objective_value + self.instance.W[job_id] * end_time
        elif objective == RootProblem.Objective.wiFi:
            return self.objective_value + self.instance.W[job_id] * (end_time - self.instance.R[job_id])
        elif objective == RootProblem.Objective.wiTi:
            return self.objective_value + self.instance.W[job_id] * max(end_time-self.instance.D[job_id],0)
        elif objective == RootProblem.Objective.Lmax:
            return max( self.objective_value, end_time-self.instance.D[job_id]) 

    def idle_time(self):
        """returns the idle time of the last machine

        Returns:
            int: idle time of the last machine
        """
        return self.machines[-1].idle_time()
    
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
            configuration_.append(Machine(int(line_content[0]), int(line_content[2]), oper_schedule=[Job(
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
                    schedule = self.machines[j].oper_schedule
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

    @staticmethod
    def random_insert(solution: FlowShopSolution, force_improve: bool = True, inplace: bool = True):
        """Performs an insert of a random job in a random position

        Args:
            solution (FlowShopSolution): Solution to be improved
            objective (RootProblem.Objective) : objective to consider
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.

        Returns:
            FlowShopSolution: New solution
        """
        if not inplace or force_improve:
            solution_copy = solution.copy()
        else:
            solution_copy = solution

        # Select the two different random jobs to be swapped 
        job_schedule = solution_copy.job_schedule
        job_schedule_len = len(job_schedule)
        old_objective = solution_copy.objective_value

        # Get the job and the position
        random_job_index = random.randrange(job_schedule_len)
        random_pos = random_job_index
        while random_pos == random_job_index:
            random_pos = random.randrange(job_schedule_len)

        print(random_job_index, random_pos)

        # Simulate applying the swap move
        random_job = job_schedule.pop(random_job_index)
        job_schedule.insert(random_pos, random_job)
        new_objective = solution_copy.compute_objective(startIndex= min(random_job_index, random_pos))

        # Update the solution
        if force_improve and (new_objective > old_objective):
            return solution

        return solution_copy

    @staticmethod
    def random_swap(solution: FlowShopSolution, force_improve: bool = True, inplace: bool = True):
        """Performs a random swap between 2 jobs

        Args:
            solution (FlowShopSolution): Solution to be improved
            objective (RootProblem.Objective) : objective to consider
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.

        Returns:
            FlowShopSolution: New solution
        """
        if not inplace or force_improve:
            solution_copy = solution.copy()
        else:
            solution_copy = solution

        # Select the two different random jobs to be swapped 
        job_schedule = solution_copy.job_schedule
        job_schedule_len = len(job_schedule)
        old_objective = solution_copy.objective_value

        random_job_index = random.randrange(job_schedule_len)
        other_job_index = random.randrange(job_schedule_len)
        while other_job_index == random_job_index:
            other_job_index = random.randrange(job_schedule_len)

        # Simulate applying the swap move
        job_schedule[random_job_index], job_schedule[other_job_index] = job_schedule[
                    other_job_index], job_schedule[random_job_index]
        
        new_objective = solution_copy.compute_objective(startIndex= min(random_job_index, other_job_index))

        # Update the solution
        if force_improve and (new_objective > old_objective):
            return solution

        return solution_copy