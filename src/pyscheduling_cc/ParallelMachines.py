from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import numpy as np
from enum import Enum
import Problem

class GenerationProtocol(Enum):
    VALLADA = 1

class GenerationLaw(Enum):
    UNIFORM = 1
    NORMAL = 2

@dataclass
class ParallelInstance(Problem.Instance,ABC):
    
    n : int # n : Number of jobs 
    m : int # m : Number of machines

    @classmethod
    @abstractmethod
    def read_txt(cls,path: Path):
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
    def generate_random(cls,protocol: str = None):
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

    def read_P(self,content : list[str],startIndex : int):
        P = []  # Matrix P_jk : Execution time of job j on machine k
        i = startIndex
        for _ in range(self.n): 
            ligne = content[i].strip().split('\t')
            P_k = [int(ligne[j]) for j in range(1,2*self.m,2) ]
            P.append(P_k)
            i += 1
        return (P,i)

    def read_R(self,content : list[str],startIndex : int):
        i = startIndex + 1
        ligne = content[i].split('\t')
        ri = [] # Table : Release time of job i
        for j in range(2, len(ligne), 2):
            ri.append(int(ligne[j]))
        return (ri,i+1)
    
    def read_S(self,content : list[str],startIndex : int):
        i = startIndex
        S = [] # Table of Matrix S_ijk : Setup time between jobs j and k on machine i
        i += 1 # Skip SSD
        while i != len(content):
            i = i+1 # Skip Mk
            Si = []
            for k in range(self.n):
                ligne = content[i].strip().split('\t')
                Sij = [int(ligne[j]) for j in range(self.n)]
                Si.append(Sij)
                i += 1
            S.append(Si)
        return (S,i)
    
    def read_D(self,content : list[str],startIndex : int):
        i = startIndex + 1
        ligne = content[i].split('\t')
        di = [] # Table : Due time of job i
        for j in range(2, len(ligne), 2):
            di.append(int(ligne[j]))
        return (di,i+1)

    def generate_P(self,protocol: GenerationProtocol,law: GenerationLaw,Pmin,Pmax):
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

    def generate_R(self,protocol: GenerationProtocol,law: GenerationLaw,PJobs : list[list[float]],Pmin : int ,Pmax : int ,alpha : float):
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

    def generate_S(self,protocol: GenerationProtocol,law: GenerationLaw,PJobs : list[list[float]],gamma : float, Smin : int = 0, Smax : int = 0):
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
                            min_p = min(PJobs[k][i],PJobs[j][i])
                            max_p = max(PJobs[k][i],PJobs[j][i])
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
        
    def generate_D(self,protocol: GenerationProtocol,law: GenerationLaw,Pmin,Pmax):
        pass

@dataclass
class Machine:

    machine_num : int
    completion_time : int = 0
    last_job : int = -1
    job_schedule : list[int] = field(default_factory=list)
    
    def __str__(self):
        return str(self.machine_num + 1) + " | " + ", ".join(map(str,self.job_schedule)) + " | " + str(self.completion_time)

    def __eq__(self,other):
        same_machine = other.machine_num == self.machine_num
        same_schedule = other.job_schedule == self.job_schedule
        return (same_machine and same_schedule)

    def copy(self):
        return Machine(self.machine_num,self.completion_time,self.last_job,list(self.job_schedule))
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    @staticmethod
    def fromDict(machine_dict):
        return Machine(machine_dict["machine_num"],machine_dict["completion_time"],machine_dict["last_job"],machine_dict["job_schedule"])

@dataclass
class ParallelSolution(Problem.Solution,ABC):

    machines_number : int
    configuration : list[Machine]

    def __init__(self,m,instance : ParallelInstance = None):
        self.machines_number = m
        self.configuration = []
        for i in range(m):
            machine = Machine(i,0,-1,[])
            self.configuration.append(machine)
        self.objective_value = 0

    def __str__(self):
        return "Objective : " + str(self.objective_value) + "\n" +"Machine_ID | Job_schedule | Completion_time\n" +  "\n".join(map(str,self.configuration))


    def copy(self):
        copy_machines = []
        for m in self.configuration:
            copy_machines.append(m.copy())
        
        copy_solution = ParallelSolution(self.m)
        for i in range(self.m):
            copy_solution.configuration[i] = copy_machines[i]
        copy_solution.objective_value = self.objective_value
        return copy_solution

    @classmethod
    @abstractmethod
    def read_txt(cls,path: Path):
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
    def method1(sol : ParallelSolution):
        pass

@dataclass
class PaarallelGA(Problem.Solver,ABC):

    @abstractmethod
    def solve(self, instance: Problem.Instance) -> Problem.SolveResult:
        """Solves the instance and returns the corresponding solve result

        Args:
            instance (Instance): instance to be solved

        Returns:
            SolveResult: object containing information about the solving process
                        and result
        """
        pass

@dataclass
class PaarallelSA(Problem.Solver,ABC):

    @abstractmethod
    def solve(self, instance: Problem.Instance) -> Problem.SolveResult:
        """Solves the instance and returns the corresponding solve result

        Args:
            instance (Instance): instance to be solved

        Returns:
            SolveResult: object containing information about the solving process
                        and result
        """
        pass