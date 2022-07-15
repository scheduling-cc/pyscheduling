from dataclasses import dataclass
from pathlib import Path
from typing import List
import ParallelMachines
@dataclass
class RmSijkCmax_Instance(ParallelMachines.ParallelInstance):
    P : List[List[int]] # Processing time
    S : List[List[List[int]]] # Setup time

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
        instance = cls("test",n,m,[],[])
        instance.P,i = instance.read_P(content,i)
        instance.S,i = instance.read_S(content,i)
        return instance

    @classmethod
    def generate_random(cls,protocol: str = None):
        """Generate a random instance according to a predefined protocol

        Args:
            protocol (string): represents the protocol used to generate the instance

        Returns:
            ParallelInstance:
        """
        pass

    def to_txt(self,path : Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+" "+str(self.m)+"\n")
        for i in range(self.n): 
            for j in range(self.m):
                f.write(str(self.P[i][j])+" ")
            f.write("\n")
        for i in range(self.m): 
            f.write("M"+str(i)+"\n")
            for j in range(self.n):
                for k in range(self.n):
                    f.write(str(self.S[i][j][k])+" ")
                f.write("\n")
        f.close()