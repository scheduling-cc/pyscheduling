from dataclasses import dataclass, field
from math import exp
from pathlib import Path
from time import perf_counter
from typing import List

import pyscheduling.JS.JobShop as JobShop
import pyscheduling.JS.JS_methods as js_methods
import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import GenerationLaw, Solver


@dataclass
class JmriwiTi_Instance(JobShop.JobShopInstance):
    P: List[List[int]] = field(default_factory=list)  # Processing time
    W: List[int] = field(default_factory=list)
    R: List[int] = field(default_factory=list)
    D: List[int] = field(default_factory=list)

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
        instance.W, i = instance.read_1D(content,i)
        instance.R, i = instance.read_1D(content,i)
        instance.D, i = instance.read_1D(content,i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, n: int, m: int, instance_name: str = "",
                        protocol: JobShop.GenerationProtocol = JobShop.GenerationProtocol.BASE, law: GenerationLaw = GenerationLaw.UNIFORM,
                        Pmin: int = 1, Pmax: int = 100,
                        Wmin: int = 1, Wmax: int = 1,
                        alpha: float = 2.0,
                        due_time_factor: float = 0.5):
        """Random generation of FmriSijkCmax problem instance
        Args:
            n (int): number of jobs of the instance
            m (int): number of machines of the instance
            protocol (FlowShop.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to FlowShop.GenerationProtocol.VALLADA.
            law (FlowShop.GenerationLaw, optional): probablistic law of generation. Defaults to FlowShop.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            Gamma (float, optional): Setup time factor. Defaults to 0.0.
            Smin (int, optional): Minimal setup time. Defaults to -1.
            Smax (int, optional): Maximal setup time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".
        Returns:
            FmSijkwiFi_Instance: the randomly generated instance
        """
        instance = cls(instance_name, n, m)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.W = instance.generate_W(protocol, law, Wmin, Wmax)
        instance.R = instance.generate_R(
                protocol, law, instance.P, Pmin, Pmax, alpha)
        instance.D = instance.generate_D(
                protocol, law, instance.P, Pmin, Pmax, due_time_factor)
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+" "+str(self.m)+"\n")

        f.write("Processing times\n")
        for job in self.P:
            for operation in job:
                f.write(str(operation[0])+"\t"+str(operation[1])+"\t")
            f.write("\n")

        f.write("Weights\n")
        for i in range(self.n):
            f.write(str(self.W[i])+"\t")

        f.write("\nRelease time\n")
        for i in range(self.n):
            f.write(str(self.R[i])+"\t")

        f.write("\nDue time\n")
        for i in range(self.n):
            f.write(str(self.D[i])+"\t")
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
        return RootProblem.Objective.wiTi

class Heuristics(js_methods.Heuristics):

    @staticmethod
    def shifting_bottleneck(instance : JmriwiTi_Instance):
        """Shifting bottleneck heuristic, Pinedo page 193

        Args:
            instance (JmCmax_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        startTime = perf_counter()
        solution = JobShop.JobShopSolution(instance)
        solution.create_solution_graph()
        jobs_completion_time = solution.graph.all_jobs_completion()
        remaining_machines = list(range(instance.m))
        scheduled_machines = []
        precedence_constraints = [] # Tuple of (job_i_id, job_j_id) with job_i preceding job_j
        K = 0.1
        criticality_rule = lambda current_jobs_completion, previous_jobs_completion : sum([instance.W[job_id]*(current_jobs_completion[job_id]-previous_jobs_completion[job_id])*exp(-max(instance.D[job_id]-current_jobs_completion[job_id],0)/K) for job_id in range(instance.n)])

        while len(remaining_machines)>0:
            machines_schedule = []
            taken_solution = None
            taken_machine = None
            edges_to_add = None
            max_criticality = None
            new_jobs_completion = None
            for machine in remaining_machines:
                
                vertices = [op[1] for op in solution.graph.get_operations_on_machine(machine)]
                job_id_mapping = {i:vertices[i] for i in range(len(vertices))}
                mapped_constraints =[]
                for precedence in precedence_constraints :
                    if precedence[0] in vertices and precedence[1] in vertices :
                        mapped_constraints.append((list(job_id_mapping.keys())
                            [list(job_id_mapping.values()).index(precedence[0])],list(job_id_mapping.keys())
                            [list(job_id_mapping.values()).index(precedence[1])]))
                rihiCi_instance = solution.graph.generate_rihiCi(machine,mapped_constraints,instance.W,instance.D,jobs_completion_time)
                
                rihiCi_solution = JobShop.rihiCi.Heuristics.ACT(rihiCi_instance).best_solution

                mapped_IDs_solution = [JobShop.Job(job_id_mapping[job.id],job.start_time,job.end_time) for job in rihiCi_solution.machine.job_schedule]
                
                temporary_edges = [((machine,mapped_IDs_solution[ind].id),(machine,mapped_IDs_solution[ind+1].id)) for ind in range(len(rihiCi_solution.machine.job_schedule)-1)]

                temporary_jobs_completion = solution.graph.temporary_job_completion(instance,temporary_edges)
                machine_criticality = criticality_rule(temporary_jobs_completion,jobs_completion_time)

                if max_criticality is None or max_criticality < machine_criticality:
                    max_criticality = machine_criticality
                    taken_solution = mapped_IDs_solution
                    taken_machine = machine
                    edges_to_add = temporary_edges
                    new_jobs_completion = temporary_jobs_completion


            remaining_machines.remove(taken_machine)
            scheduled_machines.append(taken_machine)
            # print(edges_to_add)
            jobs_completion_time = new_jobs_completion
            solution.machines[taken_machine].job_schedule = taken_solution
            solution.machines[taken_machine].objective = taken_solution[len(taken_solution)-1].end_time
            solution.graph.add_disdjunctive_arcs(instance, edges_to_add)
            precedence_constraints = list(solution.graph.generate_precedence_constraints(remaining_machines))
        
        solution.compute_objective()
        solution.objective_value = solution.graph.wiTi(instance.W,instance.D)
        

        return RootProblem.SolveResult(best_solution=solution,status=RootProblem.SolveStatus.FEASIBLE,runtime=perf_counter()-startTime,solutions=[solution])

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]