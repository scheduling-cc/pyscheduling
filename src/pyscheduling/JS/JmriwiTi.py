from dataclasses import dataclass
from math import exp
from typing import ClassVar, List

import pyscheduling.JS.JobShop as JobShop
from pyscheduling.JS.JobShop import Constraints
from pyscheduling.Problem import Objective
from pyscheduling.core.base_solvers import BaseSolver


@dataclass(init=False)
class JmriwiTi_Instance(JobShop.JobShopInstance):

    P: List[List[int]]
    W: List[int]
    R: List[int]
    D: List[int]
    S: List[List[List[int]]]
    constraints: ClassVar[Constraints] = [Constraints.P, Constraints.W, Constraints.R, Constraints.D, Constraints.S]
    objective: ClassVar[Objective] = Objective.wiTi

    @property
    def init_sol_method(self):
        return ShiftingBottleneck()


class ShiftingBottleneck(BaseSolver):

    def solve(self, instance : JmriwiTi_Instance):
        """Shifting bottleneck heuristic, Pinedo page 193

        Args:
            instance (JmCmax_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        self.notify_on_start()
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
        
        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result 
    