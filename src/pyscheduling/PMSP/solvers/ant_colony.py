import random
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.random import choice as np_choice

import pyscheduling.PMSP.ParallelMachines as pm
from pyscheduling.core.base_solvers import BaseSolver


@dataclass
class AntColony(BaseSolver):

    n_ants: int = 10
    n_best: int = 1,
    n_iterations: int = 100
    alpha : float =1
    beta : float =1
    phi: float = 0.081
    evaporation: float = 0.01
    q0: float = 0.5
    best_ants: int = 5
    pheromone_init : float =10
    instance : pm.ParallelInstance = None

    def solve(self, instance: pm.ParallelInstance):
        """Main method used to solve the problem and call the different steps
        Returns:
            SolveResult: Object containing the solution and useful metrics
        """
        self.notify_on_start()
        self.instance = instance
        self.LB = 1
        self.aco_graph = self.init_graph()
        shortest_path = None
        all_time_shortest_objective_value = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_solutions = self.gen_all_paths()
            all_solutions = self.improve_best_ants(all_solutions)

            shortest_path = min(all_solutions, key=lambda x: x[1])
            longest_path = max(all_solutions, key=lambda x: x[1])

            if shortest_path[1] == longest_path[1]:
                self.reinit_graph()

            if shortest_path[1] < all_time_shortest_objective_value[1]:
                all_time_shortest_objective_value = shortest_path
            self.spread_pheronome_global(all_solutions)

        self.notify_on_complete()

        return self.solve_result 
    
    def init_graph(self):
        """ Initialize the two stage graph with initial values of pheromone
        Returns:
            list[np.array]: list of the two stage graph consisting of np.array elements
        """
        aco_graph = []
        # Initializing pheromone
        pheromone_stage_1 = np.full(
            (self.instance.n, self.instance.m), self.pheromone_init, dtype=float)
        pheromone_stage_2 = np.full(
            (self.instance.m, self.instance.n, self.instance.n), self.pheromone_init, dtype=float)
        aco_graph.append(pheromone_stage_1)
        aco_graph.append(pheromone_stage_2)

        # Compute LB
        self.LB = self.instance.lower_bound()

        return aco_graph

    def spread_pheronome_global(self, all_solutions: List[pm.ParallelSolution]):
        """Update pheromone levels globally after finding new solutions
        Args:
            all_solutions (list[ParallelInstance_Solution]): list of generated solutions
        """
        sorted_solutions = sorted(all_solutions, key=lambda x: x[1])

        for solution, objective_value_i in sorted_solutions[:self.n_best]:
            for k in range(solution.instance.m):
                machine_k = solution.machines[k]
                for i, task_i in enumerate(machine_k.job_schedule):
                    self.aco_graph[0][task_i.id,k] += self.phi * self.LB / objective_value_i if objective_value_i > 0 else self.phi
                    
                    if i > 0:
                        prev_task = machine_k.job_schedule[i-1]
                        self.aco_graph[1][k, prev_task.id,task_i.id] += self.phi * self.LB / objective_value_i if objective_value_i > 0 else self.phi

    def improve_best_ants(self, all_solutions):
        """Apply local search to the best solution
        Args:
            all_solutions (_type_): list of all generated solutions
        Returns:
            list[RmSijkCmax_Solution]: list of updated solutions
        """
        sorted_solutions = sorted(all_solutions, key=lambda x: x[1])
        local_search = pm.PM_LocalSearch()
        for i,(solution, objective_value_i) in enumerate(sorted_solutions[:self.best_ants]):
            solution = local_search.improve(solution)
            sorted_solutions[i][1] = solution.objective_value
         
        return sorted_solutions

    def gen_all_paths(self):
        """Calls the gen_path function to generate all solutions from ants paths
        Returns:
            list[RmSijkCmax_Solution]: list of new solutions  
        """
        all_solutions = []
        for i in range(self.n_ants):
            solution_i = self.gen_path()
            if solution_i:
                self.notify_on_solution_found(solution_i)
                all_solutions.append([solution_i, solution_i.objective_value])

        return all_solutions

    def gen_path(self):
        """Generate one new solution from one ant's path, it calls the two stages: affect_tasks and sequence_tasks
        Returns:
            RmSijkCmax_Solution: new solution from ant's path
        """
        # Stage 1 : Task Affectation
        affectation = self.affect_tasks()
        for m in affectation:
            if len(m) == 0:
                return None
        # Stage 2 : Task Sequencing
        solution_path = self.sequence_tasks(affectation)

        return solution_path

    def affect_tasks(self):
        """Generates an affectation from the first stage graph and the path the ant went through
        Returns:
            list[list[int]]: List of tasks inside each machine 
        """
        pheromone = self.aco_graph[0]
        affectation = [[] for _ in range(self.instance.m)]
        for i in range(self.instance.n):
            q = random.random()
            row = (pheromone[i] ** self.alpha) * \
                ((1 / np.array(self.instance.P[i])) ** self.beta)
            row = np.nan_to_num(row)
            if row.sum() == 0:
                for j in range(self.instance.m):
                    row[j] = len(affectation[j])

                if row.sum() == 0:
                    machine = random.randrange(0, self.instance.m)
                else:
                    norm_row = row / row.sum()
                    all_inds = range(len(pheromone[i]))
                    machine = np_choice(all_inds, 1, p=norm_row)[0]

            elif q < self.q0:
                machine = np.argmax(row)
            else:
                norm_row = row / row.sum()
                all_inds = range(len(pheromone[i]))
                machine = np_choice(all_inds, 1, p=norm_row)[0]

            # Spread Pheromone Locally
            pheromone[i, machine] = (
                1-self.evaporation) * pheromone[i, machine]

            affectation[machine].append(i)
        return affectation

    def sequence_tasks(self, affectation):
        """Uses the affectation from stage 1 to sequence tasks inside machines using stage 2 of the graph
        Args:
            affectation (list[list[int]]): affectation to machines
        Returns:
            ParallelMachines.ParallelSolution: complete solution of one ant
        """
        pheromone = self.aco_graph[1]
        solution_path = pm.ParallelSolution(self.instance)

        for m in range(len(affectation)):
            machine_schedule = []
            if len(affectation[m]) > 0:
                first_task = affectation[m][random.randrange(
                    0, len(affectation[m]))]
                machine_schedule.append(pm.Job(first_task, 0, 0))
                prev = first_task

                for i in range(len(affectation[m]) - 1):
                    pheromone_i = pheromone[m, prev]
                    next_task = self.pick_task(prev, m, pheromone_i, affectation[m], [
                                               job.id for job in machine_schedule])

                    # Spread Pheromone Locally
                    pheromone_i[next_task] = (
                        1 - self.evaporation) * pheromone_i[next_task]

                    machine_schedule.append(
                        pm.Job(next_task, 0, 0))

            current_machine = solution_path.machines[m]
            current_machine.job_schedule = machine_schedule
            current_machine.compute_objective(self.instance)

        solution_path.compute_objective(self.instance)
        return solution_path

    def pick_task(self, prev, m, pheromone, affected_tasks, visited):
        """Select a task to affect according to pheromone levels and the graph's state
        Args:
            prev (int): previous segment in the graph
            m (int): number of machines
            pheromone (np.array): pheromone levels
            affected_tasks (list): list of affected tasks
            visited (list): list of visited segments
        Returns:
            int: next task to affect
        """
        pheromone_cp = np.copy(pheromone)

        pheromone_cp[:] = 0
        pheromone_cp[affected_tasks] = pheromone[affected_tasks]
        pheromone_cp[visited] = 0
        pheromone_cp[prev] = 0

        setups = np.array(self.instance.S[m][prev])
        setups[prev] = 1
        setups[visited] = 1

        q = random.random()
        if q < self.q0:
            next_task = np.argmax(
                pheromone_cp ** self.alpha * ((1.0 / setups) ** self.beta))
        else:
            row = pheromone_cp ** self.alpha * ((1.0 / setups) ** self.beta)
            row = np.nan_to_num(row)

            norm_row = row / row.sum()
            all_inds = range(self.instance.n)
            next_task = np_choice(all_inds, 1, p=norm_row)[0]

        return next_task

    def reinit_graph(self):
        """Reinitialize the graph's pheromone levels when the premature convergence is detected
        """
        r1 = random.random()
        for i in range(self.instance.n):
            for k in range(self.instance.m):
                r2 = random.random()
                if r2 < r1:
                    self.aco_graph[0][i, k] = self.pheromone_init

        r3 = random.random()
        for k in range(self.instance.m):
            for i in range(self.instance.n):
                for j in range(self.instance.n):
                    r4 = random.random()
                    if r4 < r3:
                        self.aco_graph[1][k, i, j] = self.pheromone_init