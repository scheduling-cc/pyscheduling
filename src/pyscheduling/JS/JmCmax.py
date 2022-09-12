from dataclasses import dataclass, field
from pathlib import Path
from random import randint
from time import perf_counter


import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Solver
import pyscheduling.JS.JobShop as JobShop


@dataclass
class JmCmax_Instance(JobShop.JobShopInstance):
    P: list[list[int]] = field(default_factory=list)  # Processing time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            JmCmax_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of configuration
        m = int(ligne0[1])  # number of jobs
        i = 1
        instance = cls("test", n, m)
        instance.P, i = instance.read_P(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int, configuration_number: int, protocol: JobShop.GenerationProtocol = JobShop.GenerationProtocol.VALLADA, law: JobShop.GenerationLaw = JobShop.GenerationLaw.UNIFORM, Pmin: int = -1, Pmax: int = -1, InstanceName: str = ""):
        """Random generation of JmCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            configuration_number (int): number of machines of the instance
            protocol (JobShop.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to JobShop.GenerationProtocol.VALLADA.
            law (JobShop.GenerationLaw, optional): probablistic law of generation. Defaults to JobShop.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            JmCmax_Instance: the randomly generated instance
        """
        if(Pmin == -1):
            Pmin = randint(1, 100)
        if(Pmax == -1):
            Pmax = randint(Pmin, 100)
        instance = cls(InstanceName, jobs_number, configuration_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+" "+str(self.m)+"\n")
        for job in self.P:
            for operation in job:
                f.write(str(operation[0])+"\t"+str(operation[1])+"\t")
            f.write("\n")
        f.close()

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.shifting_bottleneck

    def get_objective(self):
        """to get the objective tackled by the instance

        Returns:
            RootProblem.Objective: Makespan
        """
        return RootProblem.Objective.Cmax


class Heuristics():

    @staticmethod
    def shifting_bottleneck(instance : JmCmax_Instance):
        """Shifting bottlenech heuristic, Pinedo page 193

        Args:
            instance (JmCmax_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        solution = JobShop.JobShopSolution(instance)
        graph = JobShop.Graph(instance.P)
        Cmax = graph.critical_path()
        remaining_machines = list(range(instance.m))
        scheduled_machines = []

        while len(remaining_machines)>0:
            Cmax = graph.critical_path()
            machines_schedule = []
            taken_solution = None
            objective_value = None
            taken_machine = None
            edges_to_add = None
            for machine in remaining_machines:
                
                Lmax_instance = graph.generate_riPrecLmax(machine,Cmax)
                vertices = [op[1] for op in graph.get_operations_on_machine(machine)]
                job_id_mapping = {i:vertices[i] for i in range(len(vertices))}
                BB = JobShop.riPrecLmax.BB(Lmax_instance)
                BB.solve()
                mapped_IDs_solution = [JobShop.Job(job_id_mapping[job.id],job.start_time,job.end_time) for job in BB.best_solution]
                if objective_value is None or objective_value < BB.objective_value:
                    objective_value = BB.objective_value
                    taken_solution = mapped_IDs_solution
                    taken_machine = machine
                    edges_to_add = [((machine,mapped_IDs_solution[ind].id),(machine,mapped_IDs_solution[ind+1].id)) for ind in range(len(BB.best_solution)-1)]


            remaining_machines.remove(taken_machine)
            scheduled_machines.append(taken_machine)
            solution.machines[taken_machine].job_schedule = taken_solution
            solution.machines[taken_machine].objective = taken_solution[len(taken_solution)-1].end_time
            graph.add_disdjunctive_arcs(edges_to_add)
            solution.objective_value = graph.critical_path()

        return solution

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]