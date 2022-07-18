from dataclasses import dataclass, field
from pathlib import Path
from random import randint, uniform
import matplotlib.pyplot as plt
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
        if(Gamma == 0.0): Gamma = round(uniform(1.0, 3.0), 1)
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

@dataclass
class RmSijkCmax_Solution(ParallelMachines.ParallelSolution):

    def __init__(self,m,instance : RmSijkCmax_Instance = None, configuration : list[ParallelMachines.Machine] = None, objective_value : int = 0):
        self.instance = instance
        self.machines_number = m
        if configuration is None:
            self.configuration = []
            for i in range(m):
                machine = ParallelMachines.Machine(i,0,-1,[])
                self.configuration.append(machine)
        else: self.configuration = configuration
        self.objective_value = 0

    def __str__(self):
        return "Cmax : " + str(self.objective_value) + "\n" +"Machine_ID | Job_schedule | Completion_time\n" +  "\n".join(map(str,self.configuration))

    @classmethod
    def read_txt(cls,path: Path):
        """Read a solution from a txt file

        Args:
            path (Path): path to the solution's txt file of type Path from pathlib

        Returns:
            RmSijkCmax_Solution:
        """
        f = open(path, "r")
        content = f.read().split('\n')
        objective_value_ = int(content[0].split(':')[1])
        machines_number = len(content) - 2
        configuration_ = []
        for i in range(2,machines_number+2):
            line_content = content[i].split('|')
            configuration_.append(ParallelMachines.Machine(int(line_content[0]),int(line_content[2]),job_schedule=[int(job) for job in line_content[1].split(',')]))
        solution = cls(machines_number,objective_value=objective_value_,configuration=configuration_)
        return solution
            
    def plot(self, path : Path = None) -> None:
        """Plot the solution in an appropriate diagram"""
        # Add Tasks ID
        fig, gnt = plt.subplots() 
        
        # Setting labels for x-axis and y-axis 
        gnt.set_xlabel('seconds') 
        gnt.set_ylabel('Machines') 
        
        # Setting ticks on y-axis 

        ticks = []
        ticks_labels = []
        for i in range(self.machines_number):
            ticks.append(10*(i+1) + 5)
            ticks_labels.append(str(i+1))

        gnt.set_yticks(ticks) 
        # Labelling tickes of y-axis 
        gnt.set_yticklabels(ticks_labels) 
        
        # Setting graph attribute 
        gnt.grid(True) 
        
        for j in range(self.machines_number):
            schedule = self.configuration[j].job_schedule
            prev = -1
            prevEndTime = 0
            for element in schedule:
                job_index = element
                if prev != -1:
                    # Setup Time
                    gnt.broken_barh([(prevEndTime, self.instance.S[j][prev][job_index])], ((j+1) * 10, 8), edgecolors='black',linewidth=1, facecolors =('tab:orange'),label="Process")
                    # Processing Time
                    gnt.broken_barh([(prevEndTime + self.instance.S[j][prev][job_index], self.instance.P[job_index][j])], ((j+1) * 10, 8), edgecolors='black', linewidth=1, facecolors =('tab:blue'))
                    prevEndTime = prevEndTime + self.instance.S[j][prev][job_index] + self.instance.P[job_index][j]
                else:
                    gnt.broken_barh([(prevEndTime, self.instance.P[job_index][j])], ((j+1) * 10, 8), edgecolors='black', linewidth=1, facecolors =('tab:blue'))
                    prevEndTime = prevEndTime + self.instance.P[job_index][j]
                prev = job_index
        
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

        return

def constructive(instance : RmSijkCmax_Instance):
    solution = RmSijkCmax_Solution(instance.m)
    remaining_jobs_list = [i for i in range(instance.n)]
    while len(remaining_jobs_list) != 0:
        min_factor = None
        for i in remaining_jobs_list:
            for j in range(instance.m):
                current_machine_schedule = solution.configuration[j]
                if (current_machine_schedule.last_job == -1):
                    factor = current_machine_schedule.completion_time + instance.P[i][j]
                else:
                    factor = current_machine_schedule.completion_time + instance.P[i][j] + instance.S[j][current_machine_schedule.last_job][i]

                if not min_factor or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
                    taken_machine = j
        if (current_machine_schedule.last_job == -1):
            ci = solution.configuration[taken_machine].completion_time + instance.P[taken_job][taken_machine] 
        else:
            ci = solution.configuration[taken_machine].completion_time + instance.P[taken_job][taken_machine] + instance.S[taken_machine][solution.configuration[taken_machine].last_job][taken_job]
        solution.configuration[taken_machine].completion_time = ci
        solution.configuration[taken_machine].last_job = taken_job
        solution.configuration[taken_machine].job_schedule.append(taken_job)
        remaining_jobs_list.remove(taken_job)
        if (ci > solution.objective_value):
            solution.objective_value = ci
    return solution

instance = RmSijkCmax_Instance.generate_random(20,4)
solution = constructive(instance)
print(solution)