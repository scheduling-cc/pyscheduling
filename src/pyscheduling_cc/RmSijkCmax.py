from dataclasses import dataclass, field
from pathlib import Path
from random import randint, uniform
import matplotlib.pyplot as plt
from statistics import mean 
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
        n = int(ligne0[0]) # number of configuration
        m = int(ligne0[2]) # number of jobs
        i = 2
        instance = cls("test",n,m)
        instance.P,i = instance.read_P(content,i)
        instance.S,i = instance.read_S(content,i)
        return instance

    @classmethod
    def generate_random(cls,jobs_number : int, configuration_number : int,protocol : ParallelMachines.GenerationProtocol = ParallelMachines.GenerationProtocol.VALLADA,law : ParallelMachines.GenerationLaw = ParallelMachines.GenerationLaw.UNIFORM, Pmin : int = -1, Pmax : int = -1, Gamma : float = 0.0, Smin :  int = -1, Smax : int = -1, InstanceName : str = ""):
        if(Pmin == -1): Pmin = randint(1,100)
        if(Pmax == -1): Pmax = randint(Pmin,100)
        if(Gamma == 0.0): Gamma = round(uniform(1.0, 3.0), 1)
        if(Smin == -1): Smin = randint(1,100)
        if(Smax == -1): Smax = randint(Smin,100)
        instance = cls(InstanceName,jobs_number,configuration_number)
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
        self.configuration_number = m
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
        configuration_number = len(content) - 2
        configuration_ = []
        for i in range(2,configuration_number+2):
            line_content = content[i].split('|')
            configuration_.append(ParallelMachines.Machine(int(line_content[0]),int(line_content[2]),job_schedule=[int(job) for job in line_content[1].split(',')]))
        solution = cls(configuration_number,objective_value=objective_value_,configuration=configuration_)
        return solution
            
    def plot(self, path : Path = None) -> None:
        """Plot the solution in an appropriate diagram"""
        # Add Tasks ID
        fig, gnt = plt.subplots() 
        
        # Setting labels for x-axis and y-axis 
        gnt.set_xlabel('seconds') 
        gnt.set_ylabel('configuration') 
        
        # Setting ticks on y-axis 

        ticks = []
        ticks_labels = []
        for i in range(self.configuration_number):
            ticks.append(10*(i+1) + 5)
            ticks_labels.append(str(i+1))

        gnt.set_yticks(ticks) 
        # Labelling tickes of y-axis 
        gnt.set_yticklabels(ticks_labels) 
        
        # Setting graph attribute 
        gnt.grid(True) 
        
        for j in range(self.configuration_number):
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

class Heuristics():

    @staticmethod
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

    @staticmethod
    def list_heuristic(instance,rule=1,decreasing = False):
        solution = RmSijkCmax_Solution(instance.m)
        if rule == 1: # Mean Processings
            remaining_jobs_list = [(i,mean(instance.P[i])) for i in range(instance.n)]
        elif rule == 2: # Min Processings
            remaining_jobs_list = [(i,min(instance.P[i])) for i in range(instance.n)]
        elif rule == 3: # Mean Processings + Mean Setups
            setup_means = [mean(means_list) for means_list in [[mean(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [(i,mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif rule == 4: # Max Processings 
            remaining_jobs_list = [(i,max(instance.P[i])) for i in range(instance.n)]
        elif rule == 5: # IS1
            max_setup = [max([max(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,max(max(instance.P[i]),max_setup[i][0])) for i in range(instance.n)]
        elif rule == 6: # IS2
            min_setup = [min([min(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,max(min(instance.P[i]),min_setup[i][0])) for i in range(instance.n)]
        elif rule == 7:  # IS3
            min_setup = [min([min(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,min(min(instance.P[i]),min_setup[i][0])) for i in range(instance.n)]
        elif rule == 8: # IS4
            max_setup = [max([max(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,min(max(instance.P[i]),max_setup[i][0])) for i in range(instance.n)]
        elif rule == 9: # IS5
            max_setup = [max([max(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,max(instance.P[i])/max_setup[i][0]) for i in range(instance.n)]
        elif rule == 10: # IS6
            min_setup = [min([min(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,min(instance.P[i])/(min_setup[i][0]+1)) for i in range(instance.n)]
        elif rule == 11: # IS7
            max_setup = [max([max(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,max_setup[i][0]/max(instance.P[i])) for i in range(instance.n)]
        elif rule == 12: # IS8
            min_setup = [min([min(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,min_setup[i][0]/(min(instance.P[i])+1)) for i in range(instance.n)]
        elif rule == 13: # IS9
            min_setup = [min([min(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,min_setup[i][0]/max(instance.P[i])) for i in range(instance.n)]
        elif rule == 14: # IS10
            max_setup = [max([max(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,max_setup[i][0]/(min(instance.P[i])+1)) for i in range(instance.n)]
        elif rule == 15: # IS11
            max_setup = [max([max(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,max_setup[i][0] + max(instance.P[i])) for i in range(instance.n)]
        elif rule == 16: # IS12
            min_setup = [min([min(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,min_setup[i][0] + min(instance.P[i])) for i in range(instance.n)]
        elif rule == 17: # IS13
            proc_div_setup = [min([instance.P[i][k]/max(instance.S[k][i])] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,proc_div_setup[i]) for i in range(instance.n)]
        elif rule == 18: # IS14
            proc_div_setup = [min([instance.P[i][k]/(min(instance.S[k][i])+1)] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,proc_div_setup[i]) for i in range(instance.n)]
        elif rule == 19: # IS15
            proc_div_setup = [max([max(instance.S[k][i])/instance.P[i][k]] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,proc_div_setup[i]) for i in range(instance.n)]
        elif rule == 20: # IS16
            proc_div_setup = [max([min(instance.S[k][i])/instance.P[i][k]] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,proc_div_setup[i]) for i in range(instance.n)]
        elif rule == 21: # IS17
            proc_div_setup = [min([min(instance.S[k][i])/instance.P[i][k]] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,proc_div_setup[i]) for i in range(instance.n)]
        elif rule == 22: # IS18
            proc_div_setup = [min([max(instance.S[k][i])/instance.P[i][k]] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,proc_div_setup[i]) for i in range(instance.n)]
        elif rule == 23: # IS19
            proc_div_setup = [min([max(instance.S[k][i]) + instance.P[i][k]] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,proc_div_setup[i]) for i in range(instance.n)]
        elif rule == 24: # IS20
            proc_div_setup = [max([max(instance.S[k][i]) + instance.P[i][k]] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,proc_div_setup[i]) for i in range(instance.n)]
        elif rule == 25: # IS21
            proc_div_setup = [min([min(instance.S[k][i]) + instance.P[i][k]] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,proc_div_setup[i]) for i in range(instance.n)]
        elif rule == 26: # IS22
            proc_div_setup = [max([min(instance.S[k][i]) + instance.P[i][k]] for k in range(instance.m) )  for i in range(instance.n)]
            remaining_jobs_list = [(i,proc_div_setup[i]) for i in range(instance.n)]
        elif rule == 27: # Mean Setup
            setup_means = [mean(means_list) for means_list in [[mean(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [(i,setup_means[i]) for i in range(instance.n)]
        elif rule == 28: # Min Setup
            setup_mins = [min(min_list) for min_list in [[min(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [(i,setup_mins[i]) for i in range(instance.n)]
        elif rule == 29: # Max Setup
            setup_max = [max(max_list) for max_list in [[max(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [(i,setup_max[i]) for i in range(instance.n)]
        
        remaining_jobs_list = sorted(remaining_jobs_list,key=lambda job: job[1],reverse=decreasing)
        for element in remaining_jobs_list:
            i = element[0]
            min_factor = None
            for j in range(instance.m):
                current_machine_schedule = solution.configuration[j]
                if (current_machine_schedule.last_job == -1): # First Job
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
            if (ci > solution.objective_value):
                solution.objective_value = ci
        return solution

    @staticmethod
    def all_methods():
        return [getattr(Heuristics,func) for func in dir(Heuristics) if not func.startswith("__") and not func == "all_methods"]

instance = RmSijkCmax_Instance.generate_random(20,4)
solution = Heuristics.list_heuristic(instance)
print(solution)