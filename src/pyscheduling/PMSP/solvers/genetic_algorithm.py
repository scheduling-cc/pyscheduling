from dataclasses import dataclass
import heapq
import random
import pyscheduling.PMSP.ParallelMachines as pm
from pyscheduling.core.base_solvers import BaseSolver

from .biba import BIBA
from .grasp import GRASP
from .ordered_constructive import OrderedConstructive

@dataclass
class GeneticAlgorithm(BaseSolver):

    pop_size : int = 50
    p_cross : float = 0.7
    p_mut : float = 0.5
    p_ls : float = 1
    pressure : int = 30
    n_iterations : int = 100

    def solve(self, instance: pm.ParallelInstance):
        self.notify_on_start()
        population = self.generate_population(instance, self.pop_size,
                                            LS=(self.p_ls != 0))
        delta = 0
        i = 0
        N = 0
        local_search = pm.PM_LocalSearch()
        best_cmax = None
        best_solution = None
        solutions = []
        while i < self.n_iterations and N < 20:  # ( instance.n*instance.m*50/2000 ):
            # Select the parents
            parent_1, parent_2 = self.selection(population, self.pressure)

            # Cross parents
            pc = random.uniform(0, 1)
            if pc < self.p_cross:
                child1, child2 = self.crossover(instance, parent_1, parent_2)
            else:
                child1, child2 = parent_1, parent_2

            # Mutation
            # Child 1
            pmut = random.uniform(0, 1)
            if pmut < self.p_mut:
                child1 = self.mutation(instance, child1)
            # Child 2
            pmut = random.uniform(0, 1)
            if pmut < self.p_mut:
                child2 = self.mutation(instance, child2)

            # Local Search
            # Child 1
            pls = random.uniform(0, 1)
            if pls < self.p_ls:
                child1 = local_search.improve(child1)
            # Child 2
            pls = random.uniform(0, 1)
            if pls < self.p_ls:
                child2 = local_search.improve(child2)

            best_child = child1 if child1.objective_value <= child2.objective_value else child2
            if not best_cmax or best_child.objective_value < best_cmax:
                best_child.fix_solution()
                solutions.append(best_child)
                best_cmax = best_child.objective_value
                best_solution = best_child
                N = 0
            # Replacement
            self.replacement(population, child1)
            self.replacement(population, child2)
            i += 1
            N += 1

        self.notify_on_complete()

        return self.solve_result

    def generate_population(self, instance: pm.ParallelInstance, pop_size=40, LS=True):
        population = []
        nb_solution = 0
        heapq.heapify(population)
        local_search = pm.PM_LocalSearch()
        while nb_solution < pop_size:
            r = random.uniform(0,1)
    
            if r <= 0.2: ## Generate a solution using BIBA heuristic
                solution_i = BIBA().solve(instance).best_solution
            elif r <= 0.4: ## Generate a solution using grasp heuristic
                solution_i = GRASP(n_iterations=1).solve(instance).best_solution
            else: ## Generate a solution using ordered constructive (random)
                solution_i = OrderedConstructive(is_random = True).solve(instance).best_solution

            if LS:
                solution_i = local_search.improve(solution_i)

            self.notify_on_solution_found(solution_i)

            if solution_i not in population:
                heapq.heappush(population, solution_i)
                nb_solution += 1

        return population

    def selection(self, population, pressure):
        parent1 = population[0]
        parent2 = random.choice(population)
        
        return parent1, parent2

    def crossover(self, instance: pm.ParallelInstance, parent_1, parent_2):
        child1 = pm.ParallelSolution(instance)
        child2 = pm.ParallelSolution(instance)

        for i, machine1 in enumerate(parent_1.machines):
            machine2 = parent_2.machines[i]
            # generate 2 random crossing points
            try:
                cross_point_1 = random.randint(0, len(machine1.job_schedule)-1)
            except ValueError:
                cross_point_1 = 0
                
            try:
                cross_point_2 = random.randint(0, len(machine2.job_schedule)-1)
            except ValueError:
                cross_point_2 = 0

            child1.machines[i].job_schedule.extend(
                machine1.job_schedule[0:cross_point_1])
            child2.machines[i].job_schedule.extend(
                machine2.job_schedule[0:cross_point_2])
            
            child1.machines[i].compute_objective(instance)
            child2.machines[i].compute_objective(instance)

        self.complete_solution(instance, parent_1, child2)
        self.complete_solution(instance, parent_2, child1)

        return child1, child2

    def mutation(self, instance: pm.ParallelInstance, child: pm.ParallelSolution):
        # Random Internal Swap
        child = pm.NeighbourhoodGeneration.random_swap(
            child, force_improve=False, internal=True)
        return child

    def complete_solution(self, instance: pm.ParallelInstance, parent, child: pm.ParallelSolution):
        # Cache the jobs affected to both childs
        child_jobs = set(
            job[0] for machine in child.machines for job in machine.job_schedule)
        for i, machine_parent in enumerate(parent.machines):
            for job in machine_parent.job_schedule:
                if job[0] not in child_jobs:
                    child = pm.PM_LocalSearch.best_insertion_machine(
                        child, i, job[0])

        child.fix_objective()

    def replacement(self, population, child):
        if child not in population:
            worst_index = max(enumerate(population),
                              key=lambda x: x[1].objective_value)[0]
            if child.objective_value <= population[worst_index].objective_value:
                self.notify_on_solution_found(child)
                heapq.heappushpop(population, child)
                return True

        return False