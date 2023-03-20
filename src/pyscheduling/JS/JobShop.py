import json
import random
import sys
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import pyscheduling.Problem as RootProblem
import pyscheduling.SMSP.rihiCi as rihiCi
import pyscheduling.SMSP.riPrecLmax as riPrecLmax
from pyscheduling.Problem import GenerationLaw, Job


class GenerationProtocol(Enum):
    BASE = 1


Node = namedtuple('Node', ['machine_id', 'job_id'])

@dataclass
class JobsGraph:
    source: Node
    sink: Node
    jobs_sinks: List[Node]
    inverted_weights : bool
    DG: nx.DiGraph
    jobs_times: dict = None 

    def __init__(self, instance, invert_weights = True):
        """Create the conjunctive graph from the instance definition

        Args:
            invert_weights (bool, optional): convert the weights to negative to use shortest path algos for critical path. Defaults to True.

        Returns:
            nx.DiGraph: conjunctive graph from instance.
        """
        DG = nx.DiGraph()
        inverted_weights = -1 if invert_weights else 1
        
        
        source = Node(-2, 0)
        sink = Node(-1, -1)
        jobs_sinks = [ Node(-1, j) for j in range(instance.n) ]
        
        # Create nodes and edges
        edges_list = []
        nodes_list = [source, sink]
        nodes_list.extend(jobs_sinks)
        for j in range(instance.n):
            for nb, oper in enumerate(instance.P[j]):
                oper_node = Node(oper[0], j)
                nodes_list.append(oper_node)
                
                # Add edges
                if nb == 0: # Add source to first job edge
                    release_date = instance.R[j] if hasattr(instance, 'R') else 0 

                    edges_list.append((source, oper_node, inverted_weights * release_date))
                else: # Add precedence constraints between operations of same job (order between machines)
                    edges_list.append( (prev_oper_node, oper_node, inverted_weights * prev_oper[1]) )
                
                prev_oper_node = oper_node
                prev_oper = oper
            
            # Add last operation to sink edge
            edges_list.append((oper_node, jobs_sinks[j], inverted_weights * oper[1]))
            edges_list.append((jobs_sinks[j], sink, 0))
                
        DG.add_nodes_from(nodes_list)
        DG.add_weighted_edges_from(edges_list)
        
        self.instance = instance
        self.source = source
        self.sink = sink
        self.jobs_sinks = jobs_sinks
        self.DG = DG
        self.inverted_weights = invert_weights

    def draw(self):
        pos = nx.spring_layout(self.DG)
        nx.draw(self.DG, pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(self.DG, 'weight')
        nx.draw_networkx_edge_labels(self.DG, pos, edge_labels=edge_labels)
        plt.show()
    
    def longest_path(self, u, v):
        #return -nx.shortest_path_length(self.DG, source=u, target=v, weight='weight')
        inverted_weights = -1 if self.inverted_weights else 1
        return inverted_weights*nx.bellman_ford_path_length(self.DG, source=u, target=v, weight='weight')

    def critical_path(self):
        return self.longest_path(self.source, self.sink)

    def get_operations_on_machine(self, machine_id : int):
        """returns the vertices corresponding to operations to be executed on machine_id

        Args:
            machine_id (int): id of a machine

        Returns:
            list[tuple(int,int)]: list of operations to be executed on machine_id
        """
        return [node for node in self.DG.nodes() if node[0]==machine_id]

    def add_disdjunctive_arcs(self, instance, edges_to_add : List[tuple]):
        """Add disjunctive arcs to the graph corresponding to the operations schedule on a machine

        Args:
            edges_to_add (list[tuple(tuple(int,int),tuple(int,int))]): list of operations couples where an edge will be added from the first element of a couple to the second element of the couple
        """
        adjacency_matrix = dict(self.DG.adjacency())
        inverted_weights = -1 if self.inverted_weights else 1

        #Change the weight of the incident conjunctive edge of the first operation to include setup time
        first_op = edges_to_add[0][0]
        precedent_op = [op for op in self.DG.nodes() if self.DG.has_edge(op,first_op)][0]
        setup_time = inverted_weights*instance.S[first_op[0]][first_op[1]][first_op[1]] if hasattr(self, 'S') else 0
        self.DG[precedent_op][first_op]['weight'] = self.DG[precedent_op][first_op]['weight'] + setup_time

        for edge in edges_to_add :
            processing_time = adjacency_matrix[edge[0]][next(iter(adjacency_matrix[edge[0]]))]['weight']
            setup_time = self.S[edge[0][0]][edge[0][1]][edge[1][1]] if hasattr(self, 'S') else 0
            self.DG.add_weighted_edges_from([(edge[0],edge[1],processing_time + setup_time)])

    def generate_precedence_constraints(self, unscheduled_machines : List[int]):
        precedence_constraints = []
        for machine_id in unscheduled_machines :
            vertices = self.get_operations_on_machine(machine_id);
            for u in vertices :
                for v in vertices :
                    if u is not v and nx.has_path(self.DG, u, v) : precedence_constraints.append((u[1],v[1]))
            
        return precedence_constraints

    def generate_riPrecLmax(self, machine_id : int, Cmax : int, precedenceConstraints : List[tuple]):
        """generate an instance of 1|ri,prec|Lmax instance of the machine machine_id

        Args:
            machine_id (int): id of the machine
            Cmax (int): current makespan

        Returns:
            riPrecLmax_Instance: generated 1|ri,prec|Lmax instance
        """
        vertices = self.get_operations_on_machine(machine_id)
        jobs_number = len(vertices)

        adjacency_matrix = dict(self.DG.adjacency())
        P = [-adjacency_matrix[vertice][next(iter(adjacency_matrix[vertice]))]['weight'] for vertice in vertices]

        R = [self.longest_path(self.source,vertice) for vertice in vertices]

        D = [Cmax - self.longest_path(vertices[vertice_ind],self.sink) + P[vertice_ind]
                                    for vertice_ind in range(len(vertices))]

        return riPrecLmax.riPrecLmax_Instance(name="",n=jobs_number,P=P,R=R,D=D, Precedence=precedenceConstraints)

    def job_completion(self,job_id):
        """returns the distance of the critical path which corresponds to the Makespan

        Returns:
            int: critical path distance
        """
        return self.longest_path(self.source, self.jobs_sinks[job_id])

    def all_jobs_completion(self):
        jobs_completion = []
        for job_id in range(len(self.jobs_sinks)):
            jobs_completion.append(self.job_completion(job_id))
        return jobs_completion

    def wiTi(self, external_weights : List[int], due_dates : List[int]):
        jobs_completion = self.all_jobs_completion()
        objective_value = 0
        for job_id in range(len(jobs_completion)):
            objective_value += external_weights[job_id]*max(jobs_completion[job_id]-due_dates[job_id],0)
        return objective_value

    def temporary_job_completion(self,instance, temporary_edges : List[tuple]):
        # jobs_completion = []
        self.add_disdjunctive_arcs(instance, temporary_edges)
        # for job_id in range(len(self.sink)):
        #     jobs_completion.append(self.job_completion(job_id))
        jobs_completion = self.all_jobs_completion()
        self.DG.remove_edges_from(temporary_edges)
        return jobs_completion

    def generate_rihiCi(self, machine_id : int, precedenceConstraints : List[tuple], exeternal_weights : List[int], external_due : List[int], jobs_completion : List[int]):
        """generate an instance of 1|ri,prec|Lmax instance of the machine machine_id

        Args:
            machine_id (int): id of the machine
            Cmax (int): current makespan

        Returns:
            riPrecLmax_Instance: generated 1|ri,prec|Lmax instance
        """
        vertices = self.get_operations_on_machine(machine_id)
        jobs_number = len(vertices)
        
        adjacency_matrix = dict(self.DG.adjacency())
        P = [adjacency_matrix[vertice][next(iter(adjacency_matrix[vertice]))]['weight'] for vertice in vertices]
            
        R = [self.longest_path(self.source,vertice) for vertice in vertices]

        D = []
        for vertice_ind in range(len(vertices)) :
            Di = []
            for job_ind in range(len(self.sink)):
                distance = self.longest_path(vertices[vertice_ind],self.jobs_sinks[job_ind])
                if distance == float('-inf') : Di.append(float('inf'))
                else :
                    Di.append(max(jobs_completion[job_ind],external_due[job_ind])-distance+P[vertice_ind])
            D.append(Di)
        
        return rihiCi.rihiCi_Instance(name="", n=jobs_number, P=P, R=R, Precedence=precedenceConstraints, external_params=len(self.sink), D=D, W=exeternal_weights)


@dataclass
class JobShopInstance(RootProblem.Instance):

    n: int  # n : Number of jobs
    m: int  # m : Number of machines

    def read_P(self, content: List[str], startIndex: int):
        """Read the Processing time matrix from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the processing time matrix

        Returns:
           (list[list[int]],int): (Matrix of processing time, index of the next section of the instance)
        """
        P = []  
        i = startIndex + 1
        for _ in range(self.n):
            ligne = content[i].strip().split('\t')
            P_k = [(int(ligne[j-1]),int(ligne[j])) for j in range(1, len(ligne), 2)]
            P.append(P_k)
            i += 1
        return (P, i)

    def read_S(self, content: List[str], startIndex: int):
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
        visited_machine = list(range(self.m))
        for j in range(self.n):
            Pj = []
            nb_operation_j = random.randint(1, self.m)
            for _ in range(nb_operation_j):
                machine_id = random.randint(0, self.m-1)
                while(machine_id in [i[0] for i in Pj]) : machine_id = random.randint(0, self.m-1) # checks recirculation
                if machine_id in visited_machine: visited_machine.remove(machine_id)
                if law.name == "UNIFORM":  # Generate uniformly
                    n = int(random.uniform(Pmin, Pmax))
                elif law.name == "NORMAL":  # Use normal law
                    value = np.random.normal(0, 1)
                    n = int(abs(Pmin+Pmax*value))
                    while n < Pmin or n > Pmax:
                        value = np.random.normal(0, 1)
                        n = int(abs(Pmin+Pmax*value))
                Pj.append((machine_id,n))
            P.append(Pj)
        #If there are some unused machine by any operation
        if len(visited_machine) > 0:
            for job_list_id in range(self.n):
                for machine_id in range(job_list_id,len(visited_machine),self.n):
                    if law.name == "UNIFORM":  # Generate uniformly
                        n = int(random.uniform(Pmin, Pmax))
                    elif law.name == "NORMAL":  # Use normal law
                        value = np.random.normal(0, 1)
                        n = int(abs(Pmin+Pmax*value))
                        while n < Pmin or n > Pmax:
                            value = np.random.normal(0, 1)
                            n = int(abs(Pmin+Pmax*value))
                    P[job_list_id].append((machine_id,n))
        return P

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

    def generate_R(self, protocol: GenerationProtocol, law: GenerationLaw, PJobs: List[List[float]], Pmin: int, Pmax: int, alpha: float):
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
            sum_p = sum(oper[1] for oper in PJobs[j])
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

    def generate_S(self, protocol: GenerationProtocol, law: GenerationLaw, PJobs: List[List[float]], gamma: float, Smin: int = 0, Smax: int = 0):
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
                    if law.name == "UNIFORM":  # Use uniform law
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

    def generate_D(self, protocol: GenerationProtocol, law: GenerationLaw, PJobs: List[float], Pmin: int, Pmax: int, due_time_factor: float):
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
        PJobs = [min(oper[1] for oper in PJobs[j]) for j in range(self.n)]
        sum_p = sum(PJobs)
        for j in range(self.n):
            if hasattr(self, 'R'):
                startTime = self.R[j] + PJobs[j]
            else:
                startTime = PJobs[j]
            if law.name == "UNIFORM":  # Generate uniformly
                n = int(random.uniform(
                    startTime, startTime + due_time_factor * sum_p))

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

    machine_num: int
    objective: int = 0
    last_job: int = -1
    job_schedule: List[Job] = field(default_factory=list)

    def __init__(self, machine_num: int, objective: int = 0, last_job: int = -1, job_schedule: List[Job] = None) -> None:
        """Constructor of Machine

        Args:
            machine_num (int): ID of the machine
            objective (int, optional): completion time of the last job of the machine. Defaults to 0.
            last_job (int, optional): ID of the last job set on the machine. Defaults to -1.
            job_schedule (list[Job], optional): list of Jobs scheduled on the machine in the exact given sequence. Defaults to None.
        """
        self.machine_num = machine_num
        self.objective = objective
        self.last_job = last_job
        if job_schedule is None:
            self.job_schedule = []
        else:
            self.job_schedule = job_schedule

    def __str__(self):
        return str(self.machine_num + 1) + " | " + " : ".join(map(str, [(job.id, job.start_time, job.end_time) for job in self.job_schedule])) + " | " + str(self.objective)

    def __eq__(self, other):
        same_machine = other.machine_num == self.machine_num
        same_schedule = other.job_schedule == self.job_schedule
        return (same_machine and same_schedule)

    def copy(self):
        return Machine(self.machine_num, self.objective, self.last_job, list(self.job_schedule))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    @staticmethod
    def fromDict(machine_dict):
        return Machine(machine_dict["machine_num"], machine_dict["objective"], machine_dict["last_job"], machine_dict["job_schedule"])


@dataclass
class JobShopSolution(RootProblem.Solution):

    machines: List[Machine]

    def __init__(self, instance: JobShopInstance = None, machines: List[Machine] = None, objective_value: int = 0, graph: JobsGraph = None):
        """Constructor of RmSijkCmax_Solution

        Args:
            instance (JobShopInstance, optional): Instance to be solved by the solution. Defaults to None.
            machines (list[Machine], optional): list of machines of the instance. Defaults to None.
            objective_value (int, optional): initial objective value of the solution. Defaults to 0.
        """
        self.instance = instance
        if machines is None:
            self.machines = []
            for i in range(instance.m):
                machine = Machine(i, 0, -1, [])
                self.machines.append(machine)
        else:
            self.machines = machines
        self.objective_value = 0
        self.graph = graph
        self.job_schedule = {j: Job(j, 0, 0) for j in range(self.instance.n)}

    def _create_graph(self, invert_weights = True):
        """Create the conjunctive graph from the instance definition

        Args:
            invert_weights (bool, optional): convert the weights to negative to use shortest path algos for critical path. Defaults to True.

        Returns:
            nx.DiGraph: conjunctive graph from instance.
        """
        self.graph = JobsGraph(self.instance,invert_weights)
        return self.graph
        
    def create_solution_graph(self, invert_weights = True):
        """Create the graph containing both conjunctive and disjunctive arcs from the schedule

        Args:
            solution (JobShopSolution): the solution representing the schedule
            invert_weights (bool, optional): convert the weights to negative to use shortest path algos for critical path. Defaults to True.

        Returns:
            nx.DiGraph: graph representing the solution with a source and list of sinks
        """
        inverted_weights = -1 if invert_weights else 1
        # DF contains only conjunctive arcs
        self.graph = self._create_graph(invert_weights)
        DG = self.graph.DG

        # Add disjunctive arcs according to the schedule
        edges_list = []
        for m_id, machine in enumerate(self.machines):
            for j_idx, job in enumerate(machine.job_schedule):
                if j_idx != 0: # Add arc between prev_job and current job
                    prev_node = Node(m_id, prev_job.id)
                    curr_node = Node(m_id, job.id)
                    edges_list.append( ( prev_node, curr_node) )
                prev_job = job

        if len(edges_list)!=0 : DG.add_disdjunctive_arcs(edges_list)
        
        return self.graph

    def check_graph(self):
        """Check whether the graph is built or not yet, build it if not.
        """
        if self.graph is None:
            self.create_solution_graph()

    def is_feasible(self):
        """Check if the schedule is feasible. i.e. the graph is acyclic

        Returns:
            bool: True if the schedule is feasible
        """
        self.graph = None
        self.check_graph()
        return nx.is_directed_acyclic_graph(self.graph.DG)

    def all_completion_times(self):
        """Computes completion times from the graph using bellman ford algorithm

        Returns:
            dict: dict of completion times for each job and the makespan (-1 key)
        """
        self.check_graph()
        # Use bellman-ford to find distances from source to each sink
        nx_dists = nx.single_source_bellman_ford(self.graph.DG, source=self.graph.source)
        
        jobs_times = dict()
        for j in range(self.instance.n):
            first_machine_id = self.instance.P[j][0][0]
            start_time = nx_dists[0][Node(first_machine_id, j)]
            end_time = nx_dists[0][self.graph.jobs_sinks[j]]

            jobs_times[j] = Job(j, -start_time, -end_time)
        
        #jobs_times[-1] = Job(-1, 0, -nx_dists[0][self.graph.sink])
        self.graph.jobs_times = jobs_times
        self.job_schedule = jobs_times
        return jobs_times 

    def completion_time(self, job_id: int, recompute_distances = False):
        """Return completion time for job with job_id

        Args:
            job_id (int): id of the job
            recompute_distances (bool, optional): used to not compute distances if already computed. Defaults to False.

        Returns:
            int: completion time of job_id
        """
        if job_id not in range(-1, self.instance.n):
            return False
        
        self.check_graph()
        if recompute_distances or self.graph.jobs_times is None:
            self.all_completion_times()
        
        return self.graph.jobs_times[job_id]

    def compute_objective_graph(self, recompute_distances = False):
        """Compute the objective using the disjunctive graph. Build it if necessary

        Args:
            recompute_distances (bool, optional): used to not compute distances if already computed. Defaults to False.

        Returns:
            int: objective value
        """
        self.check_graph()
        if recompute_distances or self.graph.jobs_times is None:
            self.all_completion_times()
        
        return self.fix_objective()

    def __str__(self):
        return "Objective : " + str(self.objective_value) + "\n" + "Machine_ID | Job_schedule (job_id , start_time , completion_time) | Completion_time\n" + "\n".join(map(str, self.machines))

    def copy(self):
        copy_machines = []
        for m in self.machines:
            copy_machines.append(m.copy())

        copy_solution = JobShopSolution(self.instance)
        for i in range(self.instance.m):
            copy_solution.machines[i] = copy_machines[i]
        copy_solution.objective_value = self.objective_value
        return copy_solution

    def __lt__(self, other):
        if self.instance.get_objective().value > 0 :
            return self.objective_value < other.objective_value
        else : return other.objective_value < self.objective_value
    
    def simulate_insert_last(self, job_id : int, oper_idx : int, last_t: int):
        """returns start_time and completion_time of job_id-oper_idx if scheduled at the end of its machine's job_schedule
        Args:
            job_id (int): job to be scheduled at the end
            oper_idx (int): index of the job's operation to be scheduled
            last_t (int): latest timestamp of the job's previous operation
        Returns:
            int, int: start_time of job_id-oper_idx, completion_time of job_id-oper_idx
        """ 
        m_id, proc_time = self.instance.P[job_id][oper_idx]

        current_machine = self.machines[m_id]
        if len(current_machine.job_schedule) > 0:
            last_job, last_start, last_end = current_machine.job_schedule[-1]
        else:
            last_job, last_start, last_end = job_id, 0, 0
        # Simulate the insertion into the last position of current_machine
        release_date = self.instance.R[job_id] if hasattr(self.instance, "R") else 0
        start_time = max(release_date, last_t, last_end)
        setup_time = self.instance.S[m_id][last_job][job_id] if hasattr(self.instance, "S") else 0
        remaining_setup_time = max(setup_time-(start_time-last_t),0)

        end_time = start_time + setup_time + proc_time

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
        saved_job = self.job_schedule[job_id]
        new_obj = self.objective_value
        self.job_schedule[job_id] = Job(job_id, min(start_time, saved_job.start_time), max(end_time, saved_job.end_time))

        objective = self.instance.get_objective()
        if objective == RootProblem.Objective.Cmax:
            new_obj =  max(self.job_schedule[j].end_time for j in self.job_schedule)
        elif objective == RootProblem.Objective.wiCi:
            new_obj =  sum( self.instance.W[j] * self.job_schedule[j].end_time for j in self.job_schedule )
        elif objective == RootProblem.Objective.wiFi:
            new_obj =  sum( self.instance.W[j] * (self.job_schedule[j].end_time - self.job_schedule[j][0] ) for j in self.job_schedule )
        elif objective == RootProblem.Objective.wiTi:
            new_obj =  sum( self.instance.W[j] * max(self.job_schedule[j].end_time - self.instance.D[j], 0) for j in self.job_schedule )
        elif objective == RootProblem.Objective.Lmax:
            new_obj =  max( 0, max( self.job_schedule[j].end_time-self.instance.D[j] for j in self.job_schedule ) )
        
        self.job_schedule[job_id] = saved_job
        return new_obj

    def fix_objective(self):
        """Compute objective value of solution out of the jobs_times dict

        Args:
            jobs_times (dict): dict of job_id: (start_time, end_time)
        """
        objective = self.instance.get_objective()
        if objective == RootProblem.Objective.Cmax:
            self.objective_value = max(self.job_schedule[j].end_time for j in self.job_schedule)
        elif objective == RootProblem.Objective.wiCi:
            self.objective_value = sum( self.instance.W[j] * self.job_schedule[j].end_time for j in self.job_schedule )
        elif objective == RootProblem.Objective.wiFi:
            self.objective_value = sum( self.instance.W[j] * (self.job_schedule[j].end_time - self.job_schedule[j][0] ) for j in self.job_schedule )
        elif objective == RootProblem.Objective.wiTi:
            self.objective_value = sum( self.instance.W[j] * max(self.job_schedule[j].end_time - self.instance.D[j], 0) for j in self.job_schedule )
        elif objective == RootProblem.Objective.Lmax:
            self.objective_value = max( 0, max( self.job_schedule[j].end_time-self.instance.D[j] for j in self.job_schedule ) )

        return self.objective_value

    def compute_objective(self):
        """Compute the machines correct schedules and sets the objective value
        """
        jobs_timeline = [(0,0) for job in range(self.instance.n)] # (machine_idx_job, t)
        remaining_machines = list(range(0,self.instance.m))
        machines_timeline = {machine_id : (0,0) for machine_id in remaining_machines} # (last_job_index, t)
        jobs_times = dict()
        while len(remaining_machines) > 0 :
            for machine_id in remaining_machines :
                curr_machine = self.machines[machine_id]
                next_job_index, current_time = machines_timeline[machine_id]
                curr_job = curr_machine.job_schedule[next_job_index].id
                prev_job = curr_machine.job_schedule[next_job_index - 1].id if next_job_index > 0 else curr_job 

                while machine_id == self.instance.P[curr_job][jobs_timeline[curr_job][0]][0] :
                    
                    oper_idx, last_t = jobs_timeline[curr_job]
                    m_id, proc_time = self.instance.P[curr_job][oper_idx]

                    ri = self.instance.R[curr_job] if hasattr(self.instance, "R") else 0
                    startTime = max(current_time, last_t, ri)
                    setup_time = self.instance.S[m_id][prev_job][curr_job] if hasattr(self.instance, "S") else 0
                    endTime = startTime + setup_time + proc_time

                    new_job = Job(curr_job, startTime,endTime)
                    self.machines[machine_id].job_schedule[next_job_index] = new_job
                    
                    current_time = endTime
                    next_job_index += 1
                    prev_job = curr_job

                    jobs_timeline[curr_job] = (jobs_timeline[curr_job][0]+1, current_time)

                    global_job = jobs_times.get(curr_job, new_job)
                    jobs_times[curr_job] = Job(curr_job, min(global_job.start_time, new_job.start_time),
                                                         max(global_job.end_time, new_job.end_time) )

                    if next_job_index == len(self.machines[machine_id].job_schedule) :
                        self.machines[machine_id].objective = current_time
                        self.machines[machine_id].last_job = curr_job
                        remaining_machines.remove(machine_id)
                        break
                    else :
                        curr_job = self.machines[machine_id].job_schedule[next_job_index][0]
                machines_timeline[machine_id] = (next_job_index,current_time)
        
        self.job_schedule = jobs_times
        return self.fix_objective()

    @classmethod
    def read_txt(cls, path: Path):
        """Read a solution from a txt file

        Args:
            path (Path): path to the solution's txt file of type Path from pathlib

        Returns:
            JobShopSolution:
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

    def is_valid(self, verbosity : bool = False):
        """
        Check if solution respects the constraints
        """
        is_valid = True
        job_times = dict()
        for i in range(self.instance.n):
            ci = 0
            ri = self.instance.R[i] if hasattr(self.instance, "R") else 0
            start_i = 0
            for j, oper_tuple in enumerate(self.instance.P[i]):
                m_id, proc_time = oper_tuple
                curr_machine = self.machines[m_id]

                # Look for job i operation on machine m_id
                oper_i_m_pos = None
                for idx, job in enumerate(curr_machine.job_schedule):
                    if job.id == i: 
                        oper_i_m_pos = idx
                        break
                
                if oper_i_m_pos is None:
                    is_valid = False
                    print(f"## Error: operation {i}-{j} of job {i} on machine {m_id} is not found")
                else:
                    oper_i_m = curr_machine.job_schedule[oper_i_m_pos]
                    prev_job = curr_machine.job_schedule[oper_i_m_pos - 1].id if oper_i_m_pos > 0 else i
                    prev_m_end = curr_machine.job_schedule[oper_i_m_pos - 1].end_time if oper_i_m_pos > 0 else 0
                    setup_time = self.instance.S[m_id][prev_job][i] if hasattr(self.instance, "S") else 0

                    expected_start_time = max(ri, prev_m_end, ci)
                    expected_end_time = expected_start_time + setup_time + proc_time

                    if expected_start_time != oper_i_m.start_time or expected_end_time != oper_i_m.end_time:
                        is_valid = False
                        print(f'## Error: operation {i}-{j} expected {(i, expected_start_time, expected_end_time)}, got {oper_i_m}')
                    
                    start_i = expected_start_time if j == 0 else start_i
                    ci = expected_end_time

            job_times[i] = Job(i, start_i, ci)
        
        objective = self.instance.get_objective()
        if objective == RootProblem.Objective.Cmax:
            expected_obj =  max(job_times[j].end_time for j in range(self.instance.n))
        elif objective == RootProblem.Objective.wiCi:
            expected_obj =  sum( self.instance.W[j] * job_times[j].end_time for j in range(self.instance.n) )
        elif objective == RootProblem.Objective.wiFi:
            expected_obj =  sum( self.instance.W[j] * (job_times[j].end_time - job_times[j][0] ) for j in range(self.instance.n) )
        elif objective == RootProblem.Objective.wiTi:
            expected_obj =  sum( self.instance.W[j] * max(job_times[j].end_time - self.instance.D[j], 0) for j in range(self.instance.n) )
        elif objective == RootProblem.Objective.Lmax:
            expected_obj =  max( 0, max( job_times[j].end_time-self.instance.D[j] for j in range(self.instance.n) ) )

        if expected_obj != self.objective_value:
            is_valid = False
            print(f'## Error: objective value found {self.objective_value} expected {expected_obj}')

        return is_valid

class NeighbourhoodGeneration():

    @staticmethod
    def best_insert_oper(solution: JobShopSolution, m_id: int, job_id: int):

        curr_machine = solution.machines[m_id]
        move = None
        prev_pos = None
        for pos in range(len(curr_machine.job_schedule) + 1):
            
            if prev_pos is not None:
                curr_machine.job_schedule.pop(prev_pos)

            curr_machine.job_schedule.insert(pos, Job(job_id, 0, 0))
            if solution.is_feasible():
                new_obj = solution.compute_objective()

                if move is None or move[1] > new_obj:
                    move = (pos, new_obj)

            prev_pos = pos

        # Apply the best move
        curr_machine.job_schedule.pop(prev_pos)
        curr_machine.job_schedule.insert(move[0], Job(job_id, 0, 0))
        solution.compute_objective()

        return solution

    @staticmethod
    def random_insert(solution: JobShopSolution, force_improve: bool = False, inplace: bool = False, nb_moves: int = 1):
        """Performs an insert of a random job in a random position
        Args:
            solution (JobShopSolution): Solution to be improved
            objective (RootProblem.Objective) : objective to consider
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.
        Returns:
            JobShopSolution: New solution
        """
        if not inplace or force_improve:
            solution_copy = solution.copy()
        else:
            solution_copy = solution

        old_objective = solution_copy.objective_value

        iter = 0
        tabu_moves = set()
        while iter < nb_moves:
            # Get random machine,job and the position
            random_machine_index = random.randrange(solution.instance.m)
            job_schedule = solution_copy.machines[random_machine_index].job_schedule
            job_schedule_len = len(job_schedule)
            
            random_job_index = random.randrange(job_schedule_len)
            random_pos = random_job_index
            while random_pos == random_job_index:
                random_pos = random.randrange(job_schedule_len)
            
            if (random_machine_index, random_job_index, random_pos) in tabu_moves:
                continue

            # Simulate applying the insertion move
            random_job = job_schedule.pop(random_job_index)
            job_schedule.insert(random_pos, random_job)
            
            if solution_copy.is_feasible():
                new_objective = solution_copy.compute_objective()
                iter += 1
                tabu_moves.clear()
            else: # Add to tabu moves
                tabu_moves.add( (random_machine_index, random_job_index, random_pos) )

        # Update the solution
        if force_improve and (new_objective > old_objective):
            return solution

        return solution_copy

    @staticmethod
    def random_swap(solution: JobShopSolution, force_improve: bool = False, inplace: bool = False, nb_moves: int = 1):
        """Performs a random swap between 2 jobs
        Args:
            solution (JobShopSolution): Solution to be improved
            objective (RootProblem.Objective) : objective to consider
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.
        Returns:
            JobShopSolution: New solution
        """
        if not inplace or force_improve:
            solution_copy = solution.copy()
        else:
            solution_copy = solution

        # Select the two different random jobs to be swapped 
        old_objective = solution_copy.objective_value

        iter = 0
        tabu_moves = set()
        while iter < nb_moves:
            # Get random machine,job and the position
            random_machine_index = random.randrange(solution.instance.m)
            job_schedule = solution_copy.machines[random_machine_index].job_schedule
            job_schedule_len = len(job_schedule)

            random_job_index = random.randrange(job_schedule_len)
            other_job_index = random.randrange(job_schedule_len)
            while other_job_index == random_job_index:
                other_job_index = random.randrange(job_schedule_len)

            if (random_machine_index, random_job_index, other_job_index) in tabu_moves:
                continue
            
            # Simulate applying the swap move
            job_schedule[random_job_index], job_schedule[other_job_index] = job_schedule[
                        other_job_index], job_schedule[random_job_index]

            if solution_copy.is_feasible():
                new_objective = solution_copy.compute_objective()
                iter += 1
                tabu_moves.clear()
            else: # Add to tabu moves
                tabu_moves.add( (random_machine_index, random_job_index, other_job_index) )

        # Update the solution
        if force_improve and (new_objective > old_objective):
            return solution

        return solution_copy

    @staticmethod
    def random_neighbour(solution_i: JobShopSolution, nb_moves: int = 2):
        """Generates a random neighbour solution of the given solution
        Args:
            solution_i (JobShopSolution): Solution at iteration i
        Returns:
            JobShopSolution: New solution
        """ 
        r = random.random()
        if r < 0.5:
            solution = NeighbourhoodGeneration.random_insert(
                solution_i, force_improve=False, inplace=False, nb_moves=nb_moves)
        else:
            solution = NeighbourhoodGeneration.random_swap(
                solution_i, force_improve=False, inplace=False, nb_moves=nb_moves)
       
        return solution

    @staticmethod
    def deconstruct_construct(solution_i: JobShopSolution, d: float = 0.25):
        """Generates a random neighbour solution of the given solution using the deconstruct - construct strategy
        The procedure removes a set of jobs and insert them using best insertion (greedy) 
        Args:
            solution_i (FlowShopSolution): Solution at iteration i
        Returns:
            FlowShopSolution: New solution
        """ 
        solution_copy = solution_i.copy()
        # Deconstruction of d (percentage) random jobs out all jobs
        all_jobs = list(range(solution_copy.instance.n))
        nb_removed_jobs = int(solution_copy.instance.n * d )
        removed_jobs = random.sample(all_jobs, nb_removed_jobs)

        for machine in solution_copy.machines:
            machine.job_schedule = [job for job in machine.job_schedule if job.id not in removed_jobs]

        solution_copy.compute_objective()

        # Construction by inserting the removed jobs one by one
        for n_j, j in enumerate(removed_jobs):
            for idx, oper_tuple in enumerate(solution_copy.instance.P[j]):
                m_id, proc_time = oper_tuple
                solution_copy = NeighbourhoodGeneration.best_insert_oper(solution_copy, m_id, j)
                
        return solution_copy