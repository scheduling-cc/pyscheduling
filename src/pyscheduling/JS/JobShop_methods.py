from math import exp
import random
from time import perf_counter
from typing import Callable

import pyscheduling.Problem as RootProblem
import pyscheduling.JS.JobShop as js
from pyscheduling.Problem import Job

class Heuristics():

    @staticmethod
    def BIBA(instance: js.JobShopInstance):
        """the greedy constructive heuristic (Best insertion based approach) to find an initial solution of a Jobshop instance.

        Args:
            instance (JobShopInstance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = js.JobShopSolution(instance)
        remaining_jobs = set(range(instance.n))
        jobs_timeline = [(0,0) for i in range(instance.n)]

        while len(remaining_jobs) != 0:
            min_factor = None
            for i in remaining_jobs:
                oper_idx, last_t = jobs_timeline[i]
                m_id, proc_time = instance.P[i][oper_idx]

                start_time, end_time = solution.simulate_insert_last(i, oper_idx, last_t)
                factor = solution.simulate_insert_objective(i, start_time, end_time)
                if not min_factor or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
                    taken_machine = m_id

            curr_machine = solution.machines[taken_machine]
            curr_machine.job_schedule.append( Job(taken_job, start_time, end_time) )
            curr_machine.last_job = taken_job

            jobs_timeline[taken_job] = (jobs_timeline[taken_job][0] + 1, end_time)
            if jobs_timeline[taken_job][0] == len(instance.P[taken_job]):
                remaining_jobs.remove(taken_job)
        
        solution.compute_objective()
        return RootProblem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

    @staticmethod
    def grasp(instance: js.JobShopInstance, p: float = 0.5, r: float = 0.5, n_iterations: int = 5):
        """Returns the solution using the Greedy randomized adaptive search procedure algorithm
        Args:
            instance (SingleInstance): The instance to be solved by the heuristic
            p (float): probability of taking the greedy best solution
            r (int): percentage of moves to consider to select the best move
            nb_exec (int): Number of execution of the heuristic
        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        startTime = perf_counter()
        solveResult = RootProblem.SolveResult()
        best_solution = None

        for _ in range(n_iterations):
            solution = js.JobShopSolution(instance)
            jobs_timeline = [(0,0) for i in range(instance.n)]
            remaining_jobs = set(range(instance.n))
            while len(remaining_jobs) != 0:
                insertions_list = []
                for i in remaining_jobs:
                    oper_idx, last_t = jobs_timeline[i]
                    m_id, proc_time = instance.P[i][oper_idx]

                    start_time, end_time = solution.simulate_insert_last(i, oper_idx, last_t)
                    factor = solution.simulate_insert_objective(i, start_time, end_time)
                    insertions_list.append((i, m_id, start_time, end_time, factor))

                insertions_list.sort(key=lambda insertion: insertion[-1])
                proba = random.random()
                if proba < p:
                    rand_insertion = insertions_list[0]
                else:
                    rand_insertion = random.choice(
                        insertions_list[0:int(instance.n * r)])
                
                taken_job, m_id, start_time, end_time, factor = rand_insertion
                solution.machines[m_id].job_schedule.append( Job(taken_job, start_time, end_time) )
                solution.machines[m_id].last_job = taken_job

                jobs_timeline[taken_job] = (jobs_timeline[taken_job][0] + 1, end_time)
                if jobs_timeline[taken_job][0] == len(instance.P[taken_job]):
                    remaining_jobs.remove(taken_job)

            solution.compute_objective()
            solveResult.all_solutions.append(solution)
            if not best_solution or best_solution.objective_value > solution.objective_value:
                best_solution = solution

        solveResult.best_solution = best_solution
        solveResult.runtime = perf_counter() - startTime
        solveResult.solve_status = RootProblem.SolveStatus.FEASIBLE
        return solveResult

class Metaheuristics():


    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

