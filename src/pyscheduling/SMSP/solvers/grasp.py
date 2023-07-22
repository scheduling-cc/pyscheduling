from dataclasses import dataclass
import random
import pyscheduling.Problem as Problem
import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.SMSP.SingleMachine import Job
from pyscheduling.core.base_solvers import BaseSolver

@dataclass
class GRASP(BaseSolver):

    p : float = 0.5
    r : int = 0.5
    n_iterations : int = 5

    def solve(self, instance: SingleMachine.SingleInstance):
        """Returns the solution using the Greedy randomized adaptive search procedure algorithm

        Args:
            instance (SingleInstance): The instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        self.notify_on_start()
        for _ in range(self.n_iterations):
            solution = SingleMachine.SingleSolution(instance)
            remaining_jobs_list = [i for i in range(instance.n)]
            while len(remaining_jobs_list) != 0:
                insertions_list = []
                for i in remaining_jobs_list:
                    for k in range(0, len(solution.machine.job_schedule) + 1):
                        insertions_list.append(
                            (i, k, solution.machine.simulate_remove_insert(-1, i, k, instance)))

                insertions_list.sort(key=lambda insertion: insertion[2])
                proba = random.random()
                if proba < self.p:
                    rand_insertion = insertions_list[0]
                else:
                    rand_insertion = random.choice(
                        insertions_list[0:int(instance.n * self.r)])
                taken_job, taken_pos, ci = rand_insertion
                solution.machine.job_schedule.insert(taken_pos, Job(taken_job, 0, 0))
                solution.machine.compute_objective(instance, startIndex=taken_pos)
                solution.fix_objective()
                if taken_pos == len(solution.machine.job_schedule)-1:
                    solution.machine.last_job = taken_job
                remaining_jobs_list.remove(taken_job)

            self.notify_on_solution_found(solution)

        self.notify_on_complete()

        return self.solve_result