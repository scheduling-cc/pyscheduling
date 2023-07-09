from dataclasses import dataclass
import random

import pyscheduling.JS.JobShop as js
from pyscheduling.core.base_solvers import BaseSolver
from pyscheduling.Problem import Job

@dataclass
class GRASP(BaseSolver):

    p: float = 0.5
    r: float = 0.5 
    n_iterations: int = 5

    def solve(self, instance: js.JobShopInstance):
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
                if proba < self.p:
                    rand_insertion = insertions_list[0]
                else:
                    rand_insertion = random.choice(
                        insertions_list[0:int(instance.n * self.r)])
                
                taken_job, taken_machine, start_time, end_time, factor = rand_insertion
                # Insert taken job at the end of taken machine
                curr_machine = solution.machines[taken_machine]
                curr_machine.job_schedule.append( Job(taken_job, start_time, end_time) )
                curr_machine.last_job = taken_job

                # Update the job_schedule structure (overall schedule per job)
                _, old_start, old_end = solution.job_schedule.get(taken_job, Job(taken_job, 0, 0))
                solution.job_schedule[taken_job] = Job(taken_job, min(old_start, start_time), max(old_end, end_time))

                # Update job timeline
                oper_idx, last_t = jobs_timeline[taken_job]
                jobs_timeline[taken_job] = (oper_idx + 1, max(end_time, last_t))
                if jobs_timeline[taken_job][0] == len(instance.P[taken_job]):
                    remaining_jobs.remove(taken_job)

            solution.compute_objective()
            self.notify_on_solution_found(solution)

        self.notify_on_complete()

        return self.solve_result
    