from queue import PriorityQueue
from dataclasses import dataclass, field

@dataclass
class Graph:
    source = (-1,0)
    sink = (0,-1)

    vertices : list[tuple]
    edges : dict

    def __init__(self, operations : list[list[tuple(int, int)]]):
        self.vertices = [self.source,self.sink]
        self.edges = {}
        
        job_index = 0
        for job in operations:
            self.edges[(self.source,(job[0][0],job_index))] = 0
            nb_operation = len(job)
            for operation_ind in range(nb_operation - 1):
                self.vertices.append((job[operation_ind][0],job_index))
                self.edges[((job[operation_ind][0],job_index),(job[operation_ind+1][0],job_index))] = job[operation_ind][1]
            self.vertices.append((job[nb_operation - 1][0],job_index))
            self.edges[((job[nb_operation - 1][0],job_index),self.sink)] = job[nb_operation - 1][1]
            job_index += 1

    def add_edge(self, u : tuple(int, int), v : tuple(int, int), weight : int):
        self.edges[(u,v)] = weight

    def get_edge(self, u : tuple(int, int), v : tuple(int, int)):
        try:
            return self.edges[(u,v)]
        except:
            return -1

    def dijkstra(self, start_vertex : tuple(int, int)):
        D = {v:-float('inf') for v in self.vertices}
        D[start_vertex] = 0

        pq = PriorityQueue()
        pq.put((0, start_vertex))

        visited = []

        while not pq.empty():
            (dist, current_vertex) = pq.get()
            visited.append(current_vertex)

            for neighbor in self.vertices:
                if self.get_edge(current_vertex,neighbor) != -1:
                    distance = self.get_edge(current_vertex,neighbor)
                    if neighbor not in visited:
                        old_cost = D[neighbor]
                        new_cost = D[current_vertex] + distance
                        if new_cost > old_cost:
                            pq.put((new_cost, neighbor))
                            D[neighbor] = new_cost
        return D

    def longest_path(self,u : tuple(int, int), v : tuple(int, int)):
        return self.dijkstra(u)[v]

    def critical_path(self):
        return self.longest_path(self.source,self.sink)