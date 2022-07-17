from dataclasses import dataclass, field
from pathlib import Path
from random import randint, uniform
import ParallelMachines

@dataclass
class RmSijkCmax_Instance(ParallelMachines.ParallelInstance):
    P : list[list[int]] = field(default_factory=list)# Processing time
    S : list[list[list[int]]] = field(default_factory=list) # Setup time

    @classmethod
    def read_txt(cls,path: Path):
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
        n = int(ligne0[0]) # number of machines
        m = int(ligne0[2]) # number of jobs
        i = 2
        instance = cls("test",n,m)
        instance.P,i = instance.read_P(content,i)
        instance.S,i = instance.read_S(content,i)
        return instance

    @classmethod
    def generate_random(cls,jobs_number : int, machines_number : int,protocol : ParallelMachines.GenerationProtocol = ParallelMachines.GenerationProtocol.VALLADA,law : ParallelMachines.GenerationLaw = ParallelMachines.GenerationLaw.UNIFORM, Pmin : int = -1, Pmax : int = -1, Gamma : float = 0.0, Smin :  int = -1, Smax : int = -1, InstanceName : str = ""):
        if(Pmin == -1): Pmin = randint(1,100)
        if(Pmax == -1): Pmax = randint(Pmin,100)
        if(Gamma == 0.0): Gamma = round(uniform(33.33, 66.66), 2)
        if(Smin == -1): Smin = randint(1,100)
        if(Smax == -1): Smax = randint(Smin,100)
        instance = cls(InstanceName,jobs_number,machines_number)
        instance.P = instance.generate_P(protocol,law,Pmin,Pmax)
        instance.S = instance.generate_S(protocol,law,instance.P,Gamma,Smin,Smax)
        return instance

    def to_txt(self,path : Path) -> None:
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
        f.write("SSD\n")
        for i in range(self.m): 
            f.write("M"+str(i)+"\n")
            for j in range(self.n):
                for k in range(self.n):
                    f.write(str(self.S[i][j][k])+"\t")
                f.write("\n")
        f.close()