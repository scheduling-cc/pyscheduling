import json
import sys
import random
from queue import PriorityQueue
from enum import Enum
from pathlib import Path
from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import pyscheduling.Problem as RootProblem
import pyscheduling.SMSP.riPrecLmax as riPrecLmax
import pyscheduling.SMSP.rihiCi as rihiCi
from pyscheduling.Problem import Job

class GenerationProtocol(Enum):
    VALLADA = 1


class GenerationLaw(Enum):
    UNIFORM = 1
    NORMAL = 2

Node = namedtuple('Node', ['machine_id', 'job_id'])

@dataclass
class JobsGraph:
    source: Node
    sink: Node
    jobs_sinks: Node
    DG: nx.DiGraph
    jobs_times: dict = None 

@dataclass
class Graph:
    source = (-1,0)
    sink = (0,-1)

    vertices : list[tuple]
    edges : dict

    def __init__(self, instance):
        """Creates the dijunctives graph from a JmCmax processing times table

        Args:
            operations (list[tuple(int,int),int]): list of couples of (operation,processing time of the operation) for every job
        """
        self.vertices = [(self.source,0),(self.sink,0)]
        self.edges = {}
        
        job_index = 0
        operations = instance.P
        for job in operations:
            release_date = instance.R[job_index] if hasattr(instance, 'R') else 0 
            self.edges[(self.source,(job[0][0],job_index))] = release_date
            nb_operation = len(job)
            for operation_ind in range(nb_operation - 1):
                self.vertices.append(((job[operation_ind][0],job_index),job[operation_ind][1]))
                self.edges[((job[operation_ind][0],job_index),(job[operation_ind+1][0],job_index))] = job[operation_ind][1]
            self.vertices.append(((job[nb_operation - 1][0],job_index),job[nb_operation - 1][1]))
            self.edges[((job[nb_operation - 1][0],job_index),self.sink)] = job[nb_operation - 1][1]
            job_index += 1

    def add_edge(self, u, v, weight : int):
        """Add an edge from operation u to operation v with weight corresponding to the processing time of operation u

        Args:
            u (tuple(int,int)): operation
            v (tuple(int,int)): operation
            weight (int): processing time of operation u
        """
        self.edges[(u,v)] = weight

    def get_edge(self, u, v):
        """returns the weight of the edge from u to v

        Args:
            u (tuple(int,int)): operation
            v (tuple(int,int)): operation
            

        Returns:
            int: weight of the edge which corresponds to the processing time of operation u, is -1 if edge does not exist
        """
        try:
            return self.edges[(u,v)]
        except:
            return -1

    def get_operations_on_machine(self, machine_id : int):
        """returns the vertices corresponding to operations to be executed on machine_id

        Args:
            machine_id (int): id of a machine

        Returns:
            list[tuple(int,int)]: list of operations to be executed on machine_id
        """
        vertices = [vertice[0] for vertice in self.vertices if vertice[0][0]==machine_id]
        if machine_id==0: vertices.remove((0,-1))
        return vertices
    
    def add_disdjunctive_arcs(self, edges_to_add : list):
        """Add disjunctive arcs to the graph corresponding to the operations schedule on a machine

        Args:
            edges_to_add (list[tuple(tuple(int,int),tuple(int,int))]): list of operations couples where an edge will be added from the first element of a couple to the second element of the couple
        """
        emanating_vertices = [edge[0] for edge in edges_to_add]
        weights = []
        for emanating_vertice in emanating_vertices :
            for vertice in self.vertices :
                if vertice[0]==emanating_vertice :
                    weights.append(vertice[1])
                    break
        for edge_ind in range(len(edges_to_add)):
            self.add_edge(edges_to_add[edge_ind][0],edges_to_add[edge_ind][1],weights[edge_ind])

    def dijkstra(self, start_vertex):
        """Evaluate the longest distance from the start_vertex to every other vertex

        Args:
            start_vertex (tuple(int,int)): starting vertex

        Returns:
            dict{tuple(int,int):int}: dict where the keys are the vertices and values are the longest distance from the starting vertex to the corresponding key vertex. the value is -inf if the corresponding key vertex in unreachable from the start_vertex
        """
        vertices_list = [vertice[0] for vertice in self.vertices]
        D = {v:-float('inf') for v in vertices_list}
        D[start_vertex] = 0

        pq = PriorityQueue()
        pq.put((0, start_vertex))

        while not pq.empty():
            (dist, current_vertex) = pq.get()

            for neighbor in vertices_list:
                if self.get_edge(current_vertex,neighbor) != -1:
                    distance = self.get_edge(current_vertex,neighbor)
                    old_cost = D[neighbor]
                    new_cost = D[current_vertex] + distance
                    if new_cost > old_cost:
                        pq.put((new_cost, neighbor))
                        D[neighbor] = new_cost

        return D

    def longest_path(self,u, v):
        """returns the longest distance from vertex u to vertex v

        Args:
            u (tuple(int,int)): operation
            v (tuple(int,int)): operation

        Returns:
            int: longest distance, is -inf if v is unreachable from u
        """
        return self.dijkstra(u)[v]

    def critical_path(self):
        """returns the distance of the critical path which corresponds to the Makespan

        Returns:
            int: critical path distance
        """
        return self.longest_path(self.source,self.sink)

    def if_path(self, u, v):
        if self.get_edge(u,v) != -1 : return True
        else :
            vertices_going_to_v = [vertice[0] for vertice in self.edges.keys() if vertice[1]==v]
            for vertice in vertices_going_to_v :
                if self.if_path(u,vertice) is True : return True
            return False
        pass

    def generate_precedence_constraints(self, unscheduled_machines : list[int]):
        precedence_constraints = []
        for machine_id in unscheduled_machines :
            vertices = self.get_operations_on_machine(machine_id);
            for u in vertices :
                for v in vertices :
                    if u is not v and self.if_path(u,v) : precedence_constraints.append((u[1],v[1]))
            
        return precedence_constraints

    def generate_riPrecLmax(self, machine_id : int, Cmax : int, precedenceConstraints : list[tuple]):
        """generate an instance of 1|ri,prec|Lmax instance of the machine machine_id

        Args:
            machine_id (int): id of the machine
            Cmax (int): current makespan

        Returns:
            riPrecLmax_Instance: generated 1|ri,prec|Lmax instance
        """
        vertices = self.get_operations_on_machine(machine_id)
        jobs_number = len(vertices)
        P = []
        for vertice in vertices :
            for graph_vertice in self.vertices :
                if graph_vertice[0] == vertice : P.append(graph_vertice[1])
        R = [self.longest_path(self.source,vertice) for vertice in vertices]
        D = [Cmax - self.longest_path(vertices[vertice_ind],self.sink) + P[vertice_ind] for vertice_ind in range(len(vertices))]
        return riPrecLmax.riPrecLmax_Instance(name="",n=jobs_number,P=P,R=R,D=D, Precedence=precedenceConstraints)

@dataclass
class riwiTi_Graph:
    source = (-1,0)
    sink = list[tuple]

    vertices : list[tuple]
    edges : dict

    def __init__(self,instance):
        """Creates the dijunctives graph from a JmCmax processing times table

        Args:
            operations (list[tuple(int,int),int]): list of couples of (operation,processing time of the operation) for every job
        """
        operations = instance.P
        release_dates = instance.R
        self.vertices = [((-1,0),0)]
        self.sink = []
        for job_id in range(len(operations)): 
            self.sink.append((0,-job_id-1))
            self.vertices.append(((0,-job_id-1),0))
        self.edges = {}
        
        job_index = 0
        for job in operations:
            self.edges[(self.source,(job[0][0],job_index))] = release_dates[job_index]
            nb_operation = len(job)
            for operation_ind in range(nb_operation - 1):
                self.vertices.append(((job[operation_ind][0],job_index),job[operation_ind][1]))
                self.edges[((job[operation_ind][0],job_index),(job[operation_ind+1][0],job_index))] = job[operation_ind][1]
            self.vertices.append(((job[nb_operation - 1][0],job_index),job[nb_operation - 1][1]))
            self.edges[((job[nb_operation - 1][0],job_index),(0,-job_index-1))] = job[nb_operation - 1][1]
            job_index += 1

    def add_edge(self, u, v, weight : int):
        """Add an edge from operation u to operation v with weight corresponding to the processing time of operation u

        Args:
            u (tuple(int,int)): operation
            v (tuple(int,int)): operation
            weight (int): processing time of operation u
        """
        self.edges[(u,v)] = weight

    def get_edge(self, u, v):
        """returns the weight of the edge from u to v

        Args:
            u (tuple(int,int)): operation
            v (tuple(int,int)): operation
            

        Returns:
            int: weight of the edge which corresponds to the processing time of operation u, is -1 if edge does not exist
        """
        try:
            return self.edges[(u,v)]
        except:
            return -1

    def get_operations_on_machine(self, machine_id : int):
        """returns the vertices corresponding to operations to be executed on machine_id

        Args:
            machine_id (int): id of a machine

        Returns:
            list[tuple(int,int)]: list of operations to be executed on machine_id
        """
        vertices = [vertice[0] for vertice in self.vertices if vertice[0][0]==machine_id and vertice[0][1]>-1]
        return vertices
    
    def add_disdjunctive_arcs(self, edges_to_add : list):
        """Add disjunctive arcs to the graph corresponding to the operations schedule on a machine

        Args:
            edges_to_add (list[tuple(tuple(int,int),tuple(int,int))]): list of operations couples where an edge will be added from the first element of a couple to the second element of the couple
        """
        emanating_vertices = [edge[0] for edge in edges_to_add]
        weights = []
        for emanating_vertice in emanating_vertices :
            for vertice in self.vertices :
                if vertice[0]==emanating_vertice :
                    weights.append(vertice[1])
                    break
        for edge_ind in range(len(edges_to_add)):
            self.add_edge(edges_to_add[edge_ind][0],edges_to_add[edge_ind][1],weights[edge_ind])

    def remove_edges(self,edges_to_remove : list):
        for edge in edges_to_remove : del self.edges[edge]
    
    def dijkstra(self, start_vertex):
        """Evaluate the longest distance from the start_vertex to every other vertex

        Args:
            start_vertex (tuple(int,int)): starting vertex

        Returns:
            dict{tuple(int,int):int}: dict where the keys are the vertices and values are the longest distance from the starting vertex to the corresponding key vertex. the value is -inf if the corresponding key vertex in unreachable from the start_vertex
        """
        vertices_list = [vertice[0] for vertice in self.vertices]
        D = {v:-float('inf') for v in vertices_list}
        D[start_vertex] = 0

        pq = PriorityQueue()
        pq.put((0, start_vertex))

        while not pq.empty():
            (dist, current_vertex) = pq.get()

            for neighbor in vertices_list:
                if self.get_edge(current_vertex,neighbor) != -1:
                    distance = self.get_edge(current_vertex,neighbor)
                    old_cost = D[neighbor]
                    new_cost = D[current_vertex] + distance
                    if new_cost > old_cost:
                        pq.put((new_cost, neighbor))
                        D[neighbor] = new_cost

        return D

    def longest_path(self,u, v):
        """returns the longest distance from vertex u to vertex v

        Args:
            u (tuple(int,int)): operation
            v (tuple(int,int)): operation

        Returns:
            int: longest distance, is -inf if v is unreachable from u
        """
        return self.dijkstra(u)[v]

    def job_completion(self,job_id):
        """returns the distance of the critical path which corresponds to the Makespan

        Returns:
            int: critical path distance
        """
        return self.longest_path(self.source,self.sink[job_id])

    def all_jobs_completion(self):
        jobs_completion = []
        for job_id in range(len(self.sink)):
            jobs_completion.append(self.job_completion(job_id))
        return jobs_completion

    def wiTi(self, external_weights : list[int], due_dates : list[int]):
        jobs_completion = self.all_jobs_completion()
        objective_value = 0
        for job_id in range(len(jobs_completion)):
            objective_value += external_weights[job_id]*max(jobs_completion[job_id]-due_dates[job_id],0)
        return objective_value
    
    def temporary_job_completion(self,temporary_edges : list):
        # jobs_completion = []
        self.add_disdjunctive_arcs(temporary_edges)
        # for job_id in range(len(self.sink)):
        #     jobs_completion.append(self.job_completion(job_id))
        jobs_completion = self.all_jobs_completion()
        self.remove_edges(temporary_edges)
        return jobs_completion

    def if_path(self, u, v):
        if self.get_edge(u,v) != -1 : return True
        else :
            vertices_going_to_v = [vertice[0] for vertice in self.edges.keys() if vertice[1]==v]
            for vertice in vertices_going_to_v :
                if self.if_path(u,vertice) is True : return True
            return False
        pass

    def generate_precedence_constraints(self, unscheduled_machines : list[int]):
        precedence_constraints = []
        for machine_id in unscheduled_machines :
            vertices = self.get_operations_on_machine(machine_id);
            for u in vertices :
                for v in vertices :
                    if u is not v and self.if_path(u,v) : precedence_constraints.append((u[1],v[1]))
            
        return precedence_constraints

    def generate_rihiCi(self, machine_id : int, precedenceConstraints : list[tuple], exeternal_weights : list[int], external_due : list[int], jobs_completion : list[int]):
        """generate an instance of 1|ri,prec|Lmax instance of the machine machine_id

        Args:
            machine_id (int): id of the machine
            Cmax (int): current makespan

        Returns:
            riPrecLmax_Instance: generated 1|ri,prec|Lmax instance
        """
        vertices = self.get_operations_on_machine(machine_id)
        jobs_number = len(vertices)
        P = []
        for vertice in vertices :
            for graph_vertice in self.vertices :
                if graph_vertice[0] == vertice : P.append(graph_vertice[1])
        R = [self.longest_path(self.source,vertice) for vertice in vertices]
        D = []
        for vertice_ind in range(len(vertices)) :
            Di = []
            for job_ind in range(len(self.sink)):
                distance = self.longest_path(vertices[vertice_ind],self.sink[job_ind])
                if distance == float('-inf') : Di.append(float('inf'))
                else :
                    Di.append(max(jobs_completion[job_ind],external_due[job_ind])-distance+P[vertice_ind])
            D.append(Di)
        return rihiCi.rihiCi_Instance(name="", n=jobs_number, P=P, R=R, Precedence=precedenceConstraints, external_params=len(self.sink), D=D, W=exeternal_weights)


@dataclass
class JobShopInstance(RootProblem.Instance):

    n: int  # n : Number of jobs
    m: int  # m : Number of machines

    def create_graph(self, invert_weights = True):
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
        jobs_sinks = [ Node(-1, j) for j in range(self.n) ]
        
        # Create nodes and edges
        edges_list = []
        nodes_list = [source, sink]
        nodes_list.extend(jobs_sinks)
        for j in range(self.n):
            for nb, oper in enumerate(self.P[j]):
                oper_node = Node(oper[0], j)
                nodes_list.append(oper_node)
                
                # Add edges
                if nb == 0: # Add source to first job edge
                    release_date = self.R[j] if hasattr(self, 'R') else 0 
                    setup_time = self.S[oper[0]][j][j] if hasattr(self, 'S') else 0
                    edges_list.append((source, oper_node, inverted_weights * (release_date + setup_time)))
                else: # Add precedence constraints between operations of same job (order between machines)
                    edges_list.append( (prev_oper_node, oper_node, inverted_weights * prev_oper[1]) )
                
                prev_oper_node = oper_node
                prev_oper = oper
            
            # Add last operation to sink edge
            edges_list.append((oper_node, jobs_sinks[j], inverted_weights * oper[1]))
            edges_list.append((jobs_sinks[j], sink, 0))
                
        DG.add_nodes_from(nodes_list)
        DG.add_weighted_edges_from(edges_list)
        
        self.graph = JobsGraph(source, sink, jobs_sinks, DG)
        return self.graph

    @classmethod
    @abstractmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            JobShopInstance:

        """
        pass

    @classmethod
    @abstractmethod
    def generate_random(cls, protocol: str = None):
        """Generate a random instance according to a predefined protocol

        Args:
            protocol (string): represents the protocol used to generate the instance

        Returns:
            JobShopInstance:
        """
        pass

    @abstractmethod
    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        pass

    def read_P(self, content: list[str], startIndex: int):
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

    def read_R(self, content: list[str], startIndex: int):
        """Read the release time table from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the release time table

        Returns:
           (list[int],int): (Table of release time, index of the next section of the instance)
        """
        pass

    def read_S(self, content: list[str], startIndex: int):
        """Read the Setup time table of matrices from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the Setup time table of matrices

        Returns:
           (list[list[list[int]]],int): (Table of matrices of setup time, index of the next section of the instance)
        """
        pass

    def read_D(self, content: list[str], startIndex: int):
        """Read the due time table from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the due time table

        Returns:
           (list[int],int): (Table of due time, index of the next section of the instance)
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
        visited_machine = list(range(self.m))
        for j in range(self.n):
            Pj = []
            nb_operation_j = random.randint(1, self.m-1)
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
        pass

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
        pass

    def generate_D(self, protocol: GenerationProtocol, law: GenerationLaw, Pmin, Pmax):
        """Random generation of due time table

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            Pmin (int): Minimal processing time
            Pmax (int): Maximal processing time

        Returns:
            list[int]: due time table
        """
        pass


@dataclass
class Machine:

    machine_num: int
    objective: int = 0
    last_job: int = -1
    job_schedule: list[Job] = field(default_factory=list)

    def __init__(self, machine_num: int, objective: int = 0, last_job: int = -1, job_schedule: list[Job] = None) -> None:
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

    def compute_completion_time(self, instance: JobShopInstance, startIndex: int = 0):
        pass


@dataclass
class JobShopSolution(RootProblem.Solution):

    machines: list[Machine]

    def __init__(self, instance: JobShopInstance = None, machines: list[Machine] = None, objective_value: int = 0, graph: JobsGraph = None):
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
        self.graph = self.instance.create_graph(invert_weights)
        DG = self.graph.DG

        # Add disjunctive arcs according to the schedule
        edges_list = []
        for m_id, machine in enumerate(self.machines):
            for j_idx, job in enumerate(machine.job_schedule):
                if j_idx != 0: # Add arc between prev_job and current job
                    prev_node = Node(m_id, prev_job.id)
                    curr_node = Node(m_id, job.id)
                    
                    setup_time = self.instance.S[m_id][prev_job.id][job.id] \
                        if hasattr(self.instance, 'S') else 0
                    proc_time = [job[1] for job in self.instance.P[prev_job.id] if job[0] == m_id][0]
                    weight = inverted_weights * (proc_time + setup_time)
                    
                    edges_list.append( ( prev_node, curr_node, weight ) )
                prev_job = job
                
        DG.add_weighted_edges_from(edges_list)
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

            jobs_times[j] = (-start_time, -end_time)
        
        jobs_times[-1] = (0, -nx_dists[0][self.graph.sink])
        self.graph.jobs_times = jobs_times
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
        
        return self.fix_objective(self.graph.jobs_times)

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

        end_time = start_time + remaining_setup_time + proc_time

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
        objective = self.instance.get_objective()
        if objective == RootProblem.Objective.Cmax:
            return max(self.objective_value, end_time)
        elif objective == RootProblem.Objective.wiCi:
            return self.objective_value + self.instance.W[job_id] * end_time
        elif objective == RootProblem.Objective.wiFi:
            return self.objective_value + self.instance.W[job_id] * (end_time - self.instance.R[job_id])
        elif objective == RootProblem.Objective.wiTi:
            return self.objective_value + self.instance.W[job_id] * max(end_time-self.instance.D[job_id],0)
        elif objective == RootProblem.Objective.Lmax:
            return max( self.objective_value, end_time-self.instance.D[job_id]) 

    def fix_objective(self, jobs_times):
        """Compute objective value of solution out of the jobs_times dict

        Args:
            jobs_times (dict): dict of job_id: (start_time, end_time)
        """
        objective = self.instance.get_objective()
        if objective == RootProblem.Objective.Cmax:
            self.objective_value = max(jobs_times[j][1] for j in range(self.instance.n))
        elif objective == RootProblem.Objective.wiCi:
            self.objective_value = sum( self.instance.W[j] * jobs_times[j][1] for j in range(self.instance.n) )
        elif objective == RootProblem.Objective.wiFi:
            self.objective_value = sum( self.instance.W[j] * (jobs_times[j][1] - jobs_times[j][0] ) for j in range(self.instance.n) )
        elif objective == RootProblem.Objective.wiTi:
            self.objective_value = sum( self.instance.W[j] * max(jobs_times[j][1] - self.instance.D[j], 0) for j in range(self.instance.n) )
        elif objective == RootProblem.Objective.Lmax:
            self.objective_value = max( 0, max( jobs_times[j][1]-self.instance.D[j] for j in range(self.instance.n) ) )

        return self.objective_value

    def compute_objective(self):
        """Compute the machines correct schedules and sets the objective value
        """
        jobs_progression = [(0,0) for job in range(self.instance.n)] # (machine_idx_job, t)
        remaining_machines = list(range(0,self.instance.m))
        remaining_machines_current_job_index = {machine_id : (0,0) for machine_id in remaining_machines} # (last_job_index, t)
        while len(remaining_machines) > 0 :
            for machine_id in remaining_machines :
                current_time = remaining_machines_current_job_index[machine_id][1]
                next_job_index = remaining_machines_current_job_index[machine_id][0]
                current_job = self.machines[machine_id].job_schedule[next_job_index][0]
                while machine_id == self.instance.P[current_job][jobs_progression[current_job][0]][0] :
                    startTime = max(current_time,jobs_progression[current_job][1])
                    endTime = startTime + self.instance.P[current_job][jobs_progression[current_job][0]][1]
                    self.machines[machine_id].job_schedule[next_job_index] = Job(current_job,
                    startTime,endTime)
                    
                    current_time = endTime
                    next_job_index += 1

                    jobs_progression[current_job] = (jobs_progression[current_job][0]+1, current_time)

                    if next_job_index == len(self.machines[machine_id].job_schedule) :
                        self.machines[machine_id].objective = current_time
                        self.machines[machine_id].last_job = current_job
                        remaining_machines.remove(machine_id)
                        break
                    else :
                        current_job = self.machines[machine_id].job_schedule[next_job_index][0]
                remaining_machines_current_job_index[machine_id] = (next_job_index,current_time)
        
        jobs_times = dict()
        for j in range(self.instance.n):
            first_machine_id = self.instance.P[j][0][0]
            last_machine_id = self.instance.P[j][-1][0]

            for job in self.machines[first_machine_id].job_schedule:
                if job.id == j:
                    start_time = job.start_time
            for job in self.machines[last_machine_id].job_schedule:
                if job.id == j:
                    end_time = job.end_time
            
            jobs_times[j] = (start_time, end_time)

        return self.fix_objective(jobs_times)

    def cmax(self):
        """Sets the schedule of each machine then sets the makespan
        """
        
        if hasattr(self.instance,"R") :
            jobs_progression = [(0,self.instance.R[job]) for job in range(self.instance.n)]
        else : jobs_progression = [(0,0) for job in range(self.instance.n)]
        remaining_machines = list(range(0,self.instance.m))
        remaining_machines_current_job_index = {machine_id : (0,0) for machine_id in remaining_machines}
        while len(remaining_machines) > 0 :
            for machine_id in remaining_machines :
                current_time = remaining_machines_current_job_index[machine_id][1]
                next_job_index = remaining_machines_current_job_index[machine_id][0]
                current_job = self.machines[machine_id].job_schedule[next_job_index][0]
                while machine_id == self.instance.P[current_job][jobs_progression[current_job][0]][0] :
                    startTime = max(current_time,jobs_progression[current_job][1])
                    endTime = startTime + self.instance.P[current_job][jobs_progression[current_job][0]][1]
                    self.machines[machine_id].job_schedule[next_job_index] = Job(current_job,
                    startTime,endTime)
                    
                    current_time = endTime
                    next_job_index += 1

                    jobs_progression[current_job] = (jobs_progression[current_job][0]+1, current_time)

                    if next_job_index == len(self.machines[machine_id].job_schedule) :
                        self.machines[machine_id].objective = current_time
                        self.machines[machine_id].last_job = current_job
                        remaining_machines.remove(machine_id)
                        break
                    else :
                        current_job = self.machines[machine_id].job_schedule[next_job_index][0]
                remaining_machines_current_job_index[machine_id] = (next_job_index,current_time)

        self.objective_value = max([machine.objective for machine in self.machines])

        return self.objective_value


    def fix_cmax(self):
        pass

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
        precedence_in_jobs = [[op[0] for op in job] for job in self.instance.P]
        for machine_id in range(len(self.machines)) :
            ci = 0
            for job_ind in range(len(self.machines[machine_id].job_schedule)) :

                element = self.machines[machine_id].job_schedule[job_ind]
                job_id, start_time, end_time = element

                operation_of_job_id = precedence_in_jobs[job_id]
                job = None
                for precedence_machine_id in range(len(operation_of_job_id)) :
                    if operation_of_job_id[precedence_machine_id] == machine_id : break
                if precedence_machine_id != 0 :
                    precedent_machine_id = operation_of_job_id[precedence_machine_id-1]

                    for job in self.machines[precedent_machine_id].job_schedule :
                        if job.id == job_id : break

                    previous_op_end_time = job.end_time
                else : previous_op_end_time = 0

                
                expected_start_time = max(previous_op_end_time,ci)

                proc_time = self.instance.P[job_id][precedence_machine_id][1]
                ci = expected_start_time + proc_time

                if start_time != expected_start_time or end_time != ci:
                    if start_time > expected_start_time and end_time - start_time == proc_time:
                        if verbosity : warnings.warn(f'## Warning: found {element} could have been scheduled earlier to reduce idle time')
                    else :
                        if verbosity : 
                            print(f'## Error:  found {element} expected {job_id,expected_start_time, ci}')
                            if job is not None :
                                print(f'## {element} needs to be sheduled after {job}')
                        is_valid = False
        
        if is_valid :
            solution_copy = self.copy()
            if self.instance.get_objective() == RootProblem.Objective.wiCi:
                is_valid = self.objective_value == solution_copy.wiCi()
            elif self.instance.get_objective() == RootProblem.Objective.wiTi:
                is_valid = self.objective_value == solution_copy.wiTi()
            elif self.instance.get_objective() == RootProblem.Objective.Cmax:
                is_valid = self.objective_value == solution_copy.cmax()
            elif self.instance.get_objective() == RootProblem.Objective.Lmax:
                is_valid = self.objective_value == solution_copy.Lmax()
            if not is_valid :
                if verbosity : print(f'## Error:  objective value found {self.objective_value} expected {solution_copy.objective_value}')

        return is_valid