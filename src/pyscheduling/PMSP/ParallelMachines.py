import json
import sys
import random
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt

import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import DecoratorsHelper, Job, GenerationLaw, Constraints, Objective
from pyscheduling.SMSP.SingleMachine import Machine as SMachine

class GenerationProtocol(Enum):
    VALLADA = 1


def parallel_instance(constraints: list[Constraints], objective: Objective):
    """Decorator to build an Instance class from the list of constraints and objective

    Args:
        constraints (list[Constraints]): list of constraints of the defined problem
        objective (Objective): the objective of the defined problem
    """

    def init_fn(self, n, m, instance_name="", P=None, W=None, R=None, D=None, S=None):
        """Constructor to build an instance of the class Instance

        Args:
            n (int): number of jobs
            m (int): number of machines
            instance_name (str, optional): name of the instance. Defaults to "".
            P (list, optional): processing times vector of length n. Defaults to None.
            W (list, optional): weights vector of length n. Defaults to None.
            R (list, optional): release times vector of length n. Defaults to None.
            D (list, optional): due dates vector of length n. Defaults to None.
            S (list, optional): setup times matrix (between job i and job j) of dimension m x n x n. Defaults to None.
        """
        self.n = n
        self.m = m
        self.instance_name = instance_name
        self.P = P if P is not None else list()
        if Constraints.W in constraints:
            self.W = W if W is not None else list()
        if Constraints.R in constraints:
            self.R = R if R is not None else list()
        if Constraints.D in constraints:
            self.D = D if D is not None else list()
        if Constraints.S in constraints:
            self.S = S if S is not None else list()

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            risijCmax_Instance:
        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of jobs
        m = int(ligne0[2])  # number of machines
        i = 1
        instance = cls(n, m, "test")
        instance.P, i = instance.read_2D(n, content, i)
        if Constraints.W in constraints:
            instance.W, i = instance.read_1D(content, i)
        if Constraints.R in constraints:
            instance.R, i = instance.read_1D(content, i)
        if Constraints.D in constraints:
            instance.D, i = instance.read_1D(content, i)
        if Constraints.S in constraints:
            instance.S, i = instance.read_3D(m, n, content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int, machines_number: int, InstanceName: str = "",
                        protocol: GenerationProtocol = GenerationProtocol.VALLADA, law: GenerationLaw = GenerationLaw.UNIFORM,
                        Wmin: int = 1, Wmax: int = 1,
                        Pmin: int = 1, Pmax: int = 100,
                        alpha: float = 2.0,
                        due_time_factor: float = 0.5,
                        Gamma: float = 2.0, Smin: int = 10, Smax: int = 100):
        """Random generation of risijCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            InstanceName (str, optional): name to give to the instance. Defaults to "".
            protocol (SingleMachine.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to SingleMachine.GenerationProtocol.VALLADA.
            law (SingleMachine.GenerationLaw, optional): probablistic law of generation. Defaults to SingleMachine.GenerationLaw.UNIFORM.
            Wmin (int, optional): Minimal weight. Defaults to 1.
            Wmax (int, optional): Maximal weight. Defaults to 1.
            Pmin (int, optional): Minimal processing time. Defaults to 1.
            Pmax (int, optional): Maximal processing time. Defaults to 100.
            alpha (float, optional): Release time factor. Defaults to 2.0.
            due_time_factor (float, optional): Due time factor. Defaults to 0.5.
            Gamma (float, optional): Setup time factor. Defaults to 2.0.
            Smin (int, optional) : Minimal setup time. Defaults to 10.
            Smax (int, optional) : Maximal setup time. Defaults to 100.

        Returns:
            risijCmax_Instance: the randomly generated instance
        """
        instance = cls(jobs_number, machines_number, instance_name=InstanceName)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        if Constraints.W in constraints:
            instance.W = instance.generate_W(protocol, law, Wmin, Wmax)
        if Constraints.R in constraints:
            instance.R = instance.generate_R(
                protocol, law, instance.P, Pmin, Pmax, alpha)
        if Constraints.D in constraints:
            instance.D = instance.generate_D(
                protocol, law, instance.P, Pmin, Pmax, due_time_factor)
        if Constraints.S in constraints:
            instance.S = instance.generate_S(
                protocol, law, instance.P, Gamma, Smin, Smax)

        return instance

    def to_txt(self, path: Path):
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+"  "+str(self.m)+"\n")
        f.write(str(self.m)+"\n")
        # Processing Times
        for i in range(self.n):
            for j in range(self.m):
                f.write("\t"+str(j)+"\t"+str(self.P[i][j]))
            if i != self.n - 1:
                f.write("\n")
        
        if Constraints.W in constraints:
            f.write("\nWeights\n")
            for i in range(self.n):
                f.write(str(self.W[i])+"\t")

        if Constraints.R in constraints:
            f.write("\nRelease time\n")
            for i in range(self.n):
                f.write(str(self.R[i])+"\t")

        if Constraints.D in constraints:
            f.write("\nDue time\n")
            for i in range(self.n):
                f.write(str(self.D[i])+"\t")

        if Constraints.S in constraints:
            f.write("\nSSD\n")
            for i in range(self.m):
                f.write("M"+str(i)+"\n")
                for j in range(self.n):
                    for k in range(self.n):
                        f.write(str(self.S[i][j][k])+"\t")
                    f.write("\n")
        f.close()

    @classmethod
    def get_objective(cls):
        """to get the objective defined by the problem

        Returns:
            RootProblem.Objective: the objective passed to the decorator
        """
        return objective

    def wrap(cls):
        """Wrapper function that adds the basic Instance functions to the wrapped class

        Returns:
            Instance: subclass of Instance according to the defined problem
        """
        DecoratorsHelper.set_new_attr(cls, "__init__", init_fn)
        DecoratorsHelper.set_new_attr(cls, "__repr__", DecoratorsHelper.repr_fn)
        DecoratorsHelper.set_new_attr(cls, "__str__", DecoratorsHelper.str_fn)
        DecoratorsHelper.set_new_attr(cls, "read_txt", read_txt)
        DecoratorsHelper.set_new_attr(cls, "generate_random", generate_random)
        DecoratorsHelper.set_new_attr(cls, "to_txt", to_txt)
        DecoratorsHelper.set_new_attr(cls, "get_objective", get_objective)

        DecoratorsHelper.update_abstractmethods(cls)
        return cls

    return wrap

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
        PJobs = [min(PJobs[j]) for j in range(self.n)]
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

@dataclass
class Machine(SMachine):
    
    machine_num:int = 0 
    completion_time:int = 0
    
    def __init__(self,machine_num:int,completion_time:int = 0):
        super().__init__()
        self.machine_num = machine_num
        self.completion_time = completion_time
        return 
    
    #used in simulation of moves and compute objective
    def compute_current_ci(self, instance: ParallelInstance, prev_ci: int, job_prev_i: int, job_i: int):
        """Computes the current ci when job_i comes after job_prev_i.
        This takes into account if we have setup times and release dates.

        Args:
            instance (SingelInstance): the instance to be solved.
            prev_ci (int): the previous value of ci
            job_prev_i (int): id of the job that precedes the inserted job 
            job_i (int): id of the job to be inserted at the end

        Returns:
            tuple: (ci, start_time), the new completion time and start_time of the inserted job.
        """
        
        startTime = max(prev_ci, instance.R[job_i]) if hasattr(instance, 'R') else prev_ci
        setupTime = instance.S[self.machine_num][job_prev_i][job_i] if hasattr(instance, 'S') else 0
        proc_time = instance.P[job_i][self.machine_num]

        ci = startTime + setupTime + proc_time
        return ci, startTime
        
    #Update this method for parallel machine
    def init_cache(self, instance: ParallelInstance, startIndex: int = 0):
        """Initialize the cache if it's not defined yet

        Args:
            startIndex (int, optional): The index from which we start fixing the schedule. Defaults to 0.
            obj_cache (list[int]): The objective's cache, it can be wiCi, wiTi or other. Defaults to None.

        Returns:
            tuple: (startIndex, obj_cache) 
        """
        objective = instance.get_objective()
        obj_cache = self.objectives_map.get(objective, -1)
        if obj_cache == -1: # No cache is used
            return startIndex, None
        if obj_cache is None:  # Initialize obj_cache to the size of job_schedule
            obj_cache = [-1] * len(self.job_schedule)
            startIndex = 0
        elif len(obj_cache) != len(self.job_schedule):
            obj_cache.insert(startIndex, -1) # Insert an element in obj_cache corresponding to the position where a new job has been inserted
    
        return startIndex, obj_cache

    #used in simulation of moves and compute objective
    def compute_obj_from_ci(self, instance: ParallelInstance, ci: int, job_i: int, curr_obj: int):
        """Helper method to compute the objective value from the current ci.
        According to the objective set on the instance, the expression of the objective in function of ci changes 

        Args:
            instance (SingleInstance): the current problem instance
            ci (int): the current completion time
            job_i (int): the job that was inserted
            curr_obj (int): current objective before inserting the job (cumulative)

        Returns:
            int: obj, the new objective
        """
        objective = instance.get_objective()
        if objective == Objective.Cmax:
            return ci
        elif objective == Objective.wiCi:
            return curr_obj + instance.W[job_i] * ci
        elif objective == Objective.wiTi:
            return curr_obj + instance.W[job_i] * max(ci-instance.D[job_i], 0)
        elif objective == Objective.wiFi:
            return curr_obj + instance.W[job_i] * (ci-instance.R[job_i])
        elif objective == Objective.Lmax:
            return max(curr_obj, ci - instance.D[job_i])    
    
    #Update this method for the parallel machines
    def compute_objective(self, instance: ParallelInstance, startIndex: int = 0):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job and returns the objective

        Args:
            instance (SingleInstance): The instance associated to the machine
            startIndex (int) : The job index the function starts operating from

        Returns:
            int: objective
        """
        startIndex, obj_cache = self.init_cache(instance, startIndex)
        ci, job_prev_i, obj = self.init_obj(startIndex, obj_cache)
        
        for i in range(startIndex, len(self.job_schedule)):
            job_i = self.job_schedule[i].id
            ci, start_time = self.compute_current_ci(instance, ci, job_prev_i, job_i)
            self.job_schedule[i] = Job(job_i, start_time, ci)

            obj = self.compute_obj_from_ci(instance, ci, job_i, obj) # This is cumulative objective
            if obj_cache is not None:
                obj_cache[i] = obj
            job_prev_i = job_i

        self.objective_value = obj
        
        return obj
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
                machine = Machine(i,0)
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
    
    def compute_objective(self,objective):    
        if objective == Objective.wiTi:
            self.objective = sum([machine.objective for machine in self.machines])

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