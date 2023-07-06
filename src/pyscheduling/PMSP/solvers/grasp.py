from dataclasses import dataclass
import random
import pyscheduling.PMSP.ParallelMachines as pm
from pyscheduling.core.base_solvers import BaseSolver

@dataclass
class GRASP(BaseSolver):

    p: float = 0.5 
    r: int = 0.5 
    n_iterations: int = 5

    def solve(self, instance: pm.ParallelInstance):
        """Returns the solution using the Greedy randomized adaptive search procedure algorithm

        Args:
            instance (ParallelInstance): The instance to be solved by the metaheuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        self.notify_on_start()
        for _ in range(self.n_iterations):
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
                if proba < self.p:
                    rand_insertion = insertions_list[0]
                else:
                    rand_insertion = random.choice(insertions_list[0:int(instance.n * self.r)])
                taken_job, taken_machine, taken_pos, ci = rand_insertion

                solution.machines[taken_machine].job_schedule.insert(taken_pos, pm.Job(taken_job, 0, 0))
                solution.machines[taken_machine].compute_objective(instance, startIndex=taken_pos)
                if taken_pos == len(solution.machines[taken_machine].job_schedule)-1:
                    solution.machines[taken_machine].last_job = taken_job
                remaining_jobs_list.remove(taken_job)

            solution.fix_objective()
            self.notify_on_solution_found(solution)

        self.notify_on_complete()

        return self.solve_result