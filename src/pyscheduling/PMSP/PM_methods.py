import heapq
import random
from math import exp
from time import perf_counter
from typing import List

import numpy as np
from numpy.random import choice as np_choice

import pyscheduling.PMSP.ParallelMachines as pm
import pyscheduling.Problem as Problem
from pyscheduling.Problem import Job
from pyscheduling.core.Solver import Solver


class Heuristics():

    @staticmethod
    def ordered_constructive(instance: pm.ParallelInstance, remaining_jobs_list=None, is_random: bool = False):
        """the ordered greedy constructive heuristic to find an initial solution of RmSijkCmax problem minimalizing the factor of (processing time + setup time) of
        jobs in the given order on different machines

        Args:
            instance (pm.ParallelInstance): Instance to be solved by the heuristic
            remaining_jobs_list (list[int],optional): specific job sequence to consider by the heuristic
            is_random (bool,optional): shuffle the remaining_jobs_list if it's generated by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = pm.ParallelSolution(instance)
        if remaining_jobs_list is None:
            remaining_jobs_list = [i for i in range(instance.n)]
            if is_random:
                random.shuffle(remaining_jobs_list)
                
        for i in remaining_jobs_list:
            min_factor = None
            for j in range(instance.m):
                current_machine = solution.machines[j]
                last_pos = len(current_machine.job_schedule)
                factor = current_machine.simulate_remove_insert(-1, i, last_pos, instance)
                
                if min_factor == None or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
                    taken_machine = j
    
            curr_machine = solution.machines[taken_machine]
            last_pos = len(curr_machine.job_schedule)
            curr_machine.job_schedule.append( Job(taken_job, -1, -1) )
            curr_machine.last_job = taken_job
            curr_machine.compute_objective(instance, startIndex=last_pos)
        
        solution.fix_objective()
      
        return Problem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, all_solutions=[solution])

    @staticmethod
    def BIBA(instance: pm.ParallelInstance):
        """the greedy constructive heuristic (Best insertion based approach) to find an initial solution of a PMSP.

        Args:
            instance (ParallelInstance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = pm.ParallelSolution(instance)
        remaining_jobs_list = [i for i in range(instance.n)]
        while len(remaining_jobs_list) != 0:
            min_factor = None
            for i in remaining_jobs_list:
                for j in range(instance.m):
                    current_machine = solution.machines[j]
                    last_pos = len(current_machine.job_schedule)
                    factor = current_machine.simulate_remove_insert(-1, i, last_pos, instance)
                    if min_factor is None or (min_factor > factor):
                        min_factor = factor
                        taken_job = i
                        taken_machine = j

            curr_machine = solution.machines[taken_machine]
            last_pos = len(curr_machine.job_schedule)
            curr_machine.job_schedule.append(Job(taken_job, -1, -1) )
            curr_machine.last_job = taken_job
            curr_machine.compute_objective(instance, startIndex=last_pos)
            remaining_jobs_list.remove(taken_job)
        
        solution.fix_objective()
          
        return Problem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, all_solutions=[solution])

    @staticmethod
    def grasp(instance: pm.ParallelInstance, p: float = 0.5, r: float = 0.5, n_iterations: int = 5):
        """Returns the solution using the Greedy randomized adaptive search procedure algorithm

        Args:
            instance (ParallelInstance): The instance to be solved by the metaheuristic
            p (float): probability of taking the greedy best solution
            r (float): percentage of moves to consider to select the best move
            n_iterations (int): Number of execution of the metaheuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = Problem.SolveResult()
        solveResult.all_solutions = []
        best_solution = None
        for _ in range(n_iterations):
            solution = pm.ParallelSolution(instance)
            remaining_jobs_list = [i for i in range(instance.n)]
            while len(remaining_jobs_list) != 0:
                insertions_list = []
                for i in remaining_jobs_list:
                    for j in range(instance.m):
                        current_machine = solution.machines[j]
                        for k in range(0, len(current_machine.job_schedule) + 1):
                            insertions_list.append(
                                (i, j, k, current_machine.simulate_remove_insert(-1, i, k, instance)))

                insertions_list = sorted(insertions_list, key=lambda insertion: insertion[3])
                proba = random.random()
                if proba < p:
                    rand_insertion = insertions_list[0]
                else:
                    rand_insertion = random.choice(insertions_list[0:int(instance.n * r)])
                taken_job, taken_machine, taken_pos, ci = rand_insertion

                solution.machines[taken_machine].job_schedule.insert(taken_pos, pm.Job(taken_job, 0, 0))
                solution.machines[taken_machine].compute_objective(instance, startIndex=taken_pos)
                if taken_pos == len(solution.machines[taken_machine].job_schedule)-1:
                    solution.machines[taken_machine].last_job = taken_job
                remaining_jobs_list.remove(taken_job)

            solution.fix_objective()
            solveResult.all_solutions.append(solution)
            if best_solution is None or best_solution.objective_value > solution.objective_value:
                best_solution = solution

        solveResult.best_solution = best_solution
        solveResult.runtime = perf_counter() - startTime
        solveResult.solve_status = Problem.SolveStatus.FEASIBLE
        
        return solveResult

class Metaheuristics():

    class LAHC(Solver):

        def solve(self, instance: pm.ParallelInstance, **kwargs):
            """ Returns the solution using the LAHC algorithm

            Args:
                instance (ParallelInstance): Instance object to solve
                Lfa (int, optional): Size of the candidates list. Defaults to 25.
                n_iterations (int, optional): Number of iterations of LAHC. Defaults to 300.
                Non_improv (int, optional): LAHC stops when the number of iterations without improvement is achieved. Defaults to 50.
                LS (bool, optional): Flag to apply local search at each iteration or not. Defaults to True.
                time_limit_factor: Fixes a time limit as follows: n*m*time_limit_factor if specified, else n_iterations is taken Defaults to None
                init_sol_method: The method used to get the initial solution. Defaults to "constructive"
                seed (int, optional): Seed for the random operators to make the algo deterministic

            Returns:
                Problem.SolveResult: the solver result of the execution of the metaheuristic
            """
            # Extracting parameters
            time_limit_factor = kwargs.get("time_limit_factor", None)
            init_sol_method = kwargs.get("init_sol_method", instance.init_sol_method())
            Lfa = kwargs.get("Lfa", 30)
            n_iterations = kwargs.get("n_iterations", 500000)
            Non_improv = kwargs.get("Non_improv", 50000)
            LS = kwargs.get("LS", True)
            seed = kwargs.get("seed", None)

            if seed:
                random.seed(seed)

            first_time = perf_counter()
            if time_limit_factor:
                time_limit = instance.m * instance.n * time_limit_factor

            self.on_start()

            # Generate init solutoin using the initial solution method
            solution_init = init_sol_method(instance).best_solution
            
            if not solution_init:
                return Problem.SolveResult()

            local_search = pm.PM_LocalSearch()
            if LS: 
                solution_init = local_search.improve(solution_init)  # Improve it with LS
                solution_init.fix_solution()
                
            all_solutions = []
            solution_best = solution_init.copy()  # Save the current best solution
            all_solutions.append(solution_best)
            lahc_list = [solution_init.objective_value] * Lfa  # Create LAHC list

            N = 0
            i = 0
            current_solution = solution_init.copy()
            while i < n_iterations and N < Non_improv:
                # check time limit if exists
                if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                    break

                solution_i = pm.NeighbourhoodGeneration.lahc_neighbour(current_solution)
                solution_i.fix_solution()
                
                if LS:
                    solution_i = local_search.improve(solution_i)
                    solution_i.fix_solution()
                
                self.on_solution_found(solution_i)

                if solution_i.objective_value < current_solution.objective_value or solution_i.objective_value < lahc_list[i % Lfa]:

                    current_solution = solution_i.copy()
                    if solution_i.objective_value < solution_best.objective_value:
                        all_solutions.append(solution_i)
                        solution_best = solution_i.copy()
                        N = 0
                lahc_list[i % Lfa] = solution_i.objective_value
                i += 1
                N += 1
            
            self.on_complete()
            return self.solve_result

    @staticmethod
    def SA(instance: pm.ParallelInstance, **kwargs):
        """ Returns the solution using the simulated annealing algorithm or the restricted simulated annealing algorithm
        
        Args:
            instance (ParallelInstance): Instance object to solve
            T0 (float, optional): Initial temperature. Defaults to 1.1.
            Tf (float, optional): Final temperature. Defaults to 0.01.
            k (float, optional): Acceptance facture. Defaults to 0.1.
            b (float, optional): Cooling factor. Defaults to 0.97.
            q0 (int, optional): Probability to apply restricted swap compared to restricted insertion. Defaults to 0.5.
            n_iterations (int, optional): Number of iterations for each temperature. Defaults to 10.
            Non_improv (int, optional): SA stops when the number of iterations without improvement is achieved. Defaults to 500.
            LS (bool, optional): Flag to apply local search at each iteration or not. Defaults to True.
            time_limit_factor: Fixes a time limit as follows: n*m*time_limit_factor if specified, else n_iterations is taken Defaults to None
            init_sol_method: The method used to get the initial solution. Defaults to "constructive"
            seed (int, optional): Seed for the random operators to make the algo deterministic if fixed. Defaults to None.

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        # Extracting the parameters
        restriced = kwargs.get("restricted", False)
        time_limit_factor = kwargs.get("time_limit_factor", None)
        init_sol_method = kwargs.get("init_sol_method", instance.init_sol_method())
        T0 = kwargs.get("T0", 1.4)
        Tf = kwargs.get("Tf", 0.01) 
        k = kwargs.get("k", 0.1)
        b = kwargs.get("b", 0.99)
        q0 = kwargs.get("q0", 0.5)
        n_iterations = kwargs.get("n_iterations", 20)
        Non_improv = kwargs.get("Non_improv", 5000)
        LS = kwargs.get("LS", True)
        seed = kwargs.get("seed", None)

        if restriced:
            generationMethod = pm.NeighbourhoodGeneration.RSA_neighbour
            data = {'q0': q0}
        else:
            generationMethod = pm.NeighbourhoodGeneration.SA_neighbour
            data = {}
        if seed:
            random.seed(seed)

        first_time = perf_counter()
        if time_limit_factor:
            time_limit = instance.m * instance.n * time_limit_factor

        solution_init = init_sol_method(instance).best_solution

        if not solution_init:
            return Problem.SolveResult()

        local_search = pm.PM_LocalSearch()

        if LS:
            solution_init = local_search.improve(solution_init)
            solution_init.fix_solution()

        all_solutions = []
        # Initialisation
        T = T0
        N = 0
        time_to_best = 0
        solution_i = None
        all_solutions.append(solution_init)
        solution_best = solution_init.copy()
        while T > Tf and (N != Non_improv):
            # check time limit if exists
            if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                break
            for i in range(0, n_iterations):
                # check time limit if exists
                if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                    break

                # solution_i = ParallelMachines.NeighbourhoodGeneration.generate_NX(solution_best)  # Generate solution in Neighbour
                solution_i = generationMethod(solution_best, **data)
                solution_i.fix_solution()
                if LS:
                    # Improve generated solution using LS
                    solution_i = local_search.improve(solution_i)
                    solution_i.fix_solution()

                delta_cmax = solution_init.objective_value - solution_i.objective_value
                if delta_cmax >= 0:
                    solution_init = solution_i.copy()
                else:
                    r = random.random()
                    factor = delta_cmax / (k * T)
                    exponent = exp(factor)
                    if (r < exponent):
                        solution_init = solution_i.copy()

                if solution_best.objective_value > solution_init.objective_value:
                    all_solutions.append(solution_init)
                    solution_best = solution_init.copy()
                    time_to_best = (perf_counter() - first_time)
                    N = 0

            T = T * b
            N += 1
            
        # Construct the solve result
        solve_result = Problem.SolveResult(
            best_solution=solution_best,
            runtime=(perf_counter() - first_time),
            time_to_best=time_to_best,
            all_solutions=all_solutions
        )
    
        return solve_result
    
    @staticmethod
    def GA(instance: pm.ParallelInstance, **kwargs):
        """Returns the solution using the genetic algorithm

        Args:
            instance (pm.ParallelInstance): Instance to be solved

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = Problem.SolveResult()
        solveResult.best_solution, solveResult.all_solutions = GeneticAlgorithm.solve(
            instance, **kwargs) 
        solveResult.solve_status = Problem.SolveStatus.FEASIBLE
        solveResult.runtime = perf_counter() - startTime
        return solveResult
    
    @staticmethod
    def antColony(instance: pm.ParallelInstance, **data):
        """Returns the solution using the ant colony algorithm

        Args:
            instance (ParallelInstance): Instance to be solved

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = Problem.SolveResult()
        AC = AntColony(instance=instance, **data)
        solveResult.best_solution, solveResult.all_solutions = AC.solve() 
        solveResult.solve_status = Problem.SolveStatus.FEASIBLE
        solveResult.runtime = perf_counter() - startTime
        return solveResult

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

class GeneticAlgorithm():

    @staticmethod
    def solve(instance: pm.ParallelInstance, pop_size=50, p_cross=0.7, p_mut=0.5, p_ls=1, pressure=30, n_iterations=100):
        population = GeneticAlgorithm.generate_population(
            instance, pop_size, LS=(p_ls != 0))
        delta = 0
        i = 0
        N = 0
        local_search = pm.PM_LocalSearch()
        best_cmax = None
        best_solution = None
        solutions = []
        while i < n_iterations and N < 20:  # ( instance.n*instance.m*50/2000 ):
            # Select the parents
            parent_1, parent_2 = GeneticAlgorithm.selection(population, pressure)

            # Cross parents
            pc = random.uniform(0, 1)
            if pc < p_cross:
                child1, child2 = GeneticAlgorithm.crossover(instance, parent_1, parent_2)
            else:
                child1, child2 = parent_1, parent_2

            # Mutation
            # Child 1
            pmut = random.uniform(0, 1)
            if pmut < p_mut:
                child1 = GeneticAlgorithm.mutation(instance, child1)
            # Child 2
            pmut = random.uniform(0, 1)
            if pmut < p_mut:
                child2 = GeneticAlgorithm.mutation(instance, child2)

            # Local Search
            # Child 1
            pls = random.uniform(0, 1)
            if pls < p_ls:
                child1 = local_search.improve(child1)
            # Child 2
            pls = random.uniform(0, 1)
            if pls < p_ls:
                child2 = local_search.improve(child2)

            best_child = child1 if child1.objective_value <= child2.objective_value else child2
            if not best_cmax or best_child.objective_value < best_cmax:
                best_child.fix_solution()
                solutions.append(best_child)
                best_cmax = best_child.objective_value
                best_solution = best_child
                N = 0
            # Replacement
            GeneticAlgorithm.replacement(population, child1)
            GeneticAlgorithm.replacement(population, child2)
            i += 1
            N += 1
       
        return best_solution, solutions

    @staticmethod
    def generate_population(instance: pm.ParallelInstance, pop_size=40, LS=True):
        population = []
        nb_solution = 0
        heapq.heapify(population)
        local_search = pm.PM_LocalSearch()
        while nb_solution < pop_size:
            r = random.uniform(0,1)
    
            if r <= 0.2: ## Generate a solution using BIBA heuristic
                solution_i = Heuristics.BIBA(instance).best_solution
            elif r <= 0.2: ## Generate a solution using grasp heuristic
                solution_i = Heuristics.grasp(instance).best_solution
            else: ## Generate a solution using ordered constructive (random)
                solution_i = Heuristics.ordered_constructive(
                    instance, **{"is_random": True}).best_solution

            if LS:
                solution_i = local_search.improve(solution_i)

            if solution_i not in population:
                heapq.heappush(population, solution_i)
                nb_solution += 1

        return population

    @staticmethod
    def selection(population, pressure):
        parent1 = population[0]
        parent2 = random.choice(population)
        
        return parent1, parent2

    @staticmethod
    def crossover(instance: pm.ParallelInstance, parent_1, parent_2):
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

        GeneticAlgorithm.complete_solution(instance, parent_1, child2)
        GeneticAlgorithm.complete_solution(instance, parent_2, child1)

        return child1, child2

    @staticmethod
    def mutation(instance: pm.ParallelInstance, child: pm.ParallelSolution):
        # Random Internal Swap
        child = pm.NeighbourhoodGeneration.random_swap(
            child, force_improve=False, internal=True)
        return child

    @staticmethod
    def complete_solution(instance: pm.ParallelInstance, parent, child: pm.ParallelSolution):
        # Cache the jobs affected to both childs
        child_jobs = set(
            job[0] for machine in child.machines for job in machine.job_schedule)
        for i, machine_parent in enumerate(parent.machines):
            for job in machine_parent.job_schedule:
                if job[0] not in child_jobs:
                    child = pm.PM_LocalSearch.best_insertion_machine(
                        child, i, job[0])

        child.fix_objective()

    @staticmethod
    def replacement(population, child):
        if child not in population:
            worst_index = max(enumerate(population),
                              key=lambda x: x[1].objective_value)[0]
            if child.objective_value <= population[worst_index].objective_value:
                heapq.heappushpop(population, child)
                return True

        return False

class AntColony(object):

    def __init__(self, instance: pm.ParallelInstance, n_ants: int = 10, n_best: int = 1,
                 n_iterations: int = 100, alpha=1, beta=1, phi: float = 0.081, evaporation: float = 0.01,
                 q0: float = 0.5, best_ants: int = 10, pheromone_init=10):
        
        """_summary_
            Args:
                distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
                n_ants (int): Number of ants running per iteration
                n_best (int): Number of best ants who deposit pheromone
                n_iteration (int): Number of iterations
                decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
                alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
                beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
            Example:
                ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)  
        """
        self.instance = instance
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.evaporation = evaporation
        self.q0 = q0
        self.best_ants = best_ants
        self.pheromone_init = pheromone_init
        self.LB = 1
        self.aco_graph = self.init_graph()

    def solve(self):
        """Main method used to solve the problem and call the different steps
        Returns:
            SolveResult: Object containing the solution and useful metrics
        """
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

        return all_time_shortest_objective_value[0], [solution[0] for solution in all_solutions]

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