from dataclasses import dataclass
import random

import pyscheduling.FS.FlowShop as FS
from pyscheduling.Problem import Job
from pyscheduling.core.base_solvers import BaseSolver


@dataclass
class GRASP(BaseSolver):

    p: float = 0.5 
    r: int = 0.5 
    n_iterations: int = 5

    def solve(self, instance: FS.FlowShopInstance):
        """Returns the solution using the Greedy randomized adaptive search procedure algorithm

        Args:
            instance (SingleInstance): The instance to be solved by the heuristic
            p (float): probability of taking the greedy best solution
            r (int): percentage of moves to consider to select the best move
            nb_exec (int): Number of execution of the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        self.notify_on_start()
        for _ in range(self.n_iterations):
            solution = FS.FlowShopSolution(instance)
            remaining_jobs_list = [i for i in range(instance.n)]
            while len(remaining_jobs_list) != 0:
                insertions_list = []
                for i in remaining_jobs_list:
                    start_time, end_time = solution.simulate_insert_last(i)
                    new_obj = solution.simulate_insert_objective(i, start_time, end_time)
                    insertions_list.append((i, new_obj))

                insertions_list.sort(key=lambda insertion: insertion[1])
                proba = random.random()
                if proba < self.p:
                    rand_insertion = insertions_list[0]
                else:
                    rand_insertion = random.choice(
                        insertions_list[0:int(instance.n * self.r)])
                
                taken_job, new_obj = rand_insertion
                solution.job_schedule.append(Job(taken_job, 0, 0))
                solution.compute_objective(startIndex=len(solution.job_schedule) - 1)
                remaining_jobs_list.remove(taken_job)

            self.notify_on_solution_found(solution)

        self.notify_on_complete()
        
        return self.solve_result

    