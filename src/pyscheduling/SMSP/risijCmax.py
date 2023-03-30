from dataclasses import dataclass
from time import perf_counter
from typing import ClassVar, List

import pyscheduling.Problem as Problem
import pyscheduling.SMSP.SingleMachine as SingleMachine
import pyscheduling.SMSP.SM_methods as Methods
from pyscheduling.Problem import Objective
from pyscheduling.SMSP.SingleMachine import Constraints
from pyscheduling.SMSP.SM_methods import ExactSolvers

@dataclass(init=False)
class risijCmax_Instance(SingleMachine.SingleInstance):
    
    P: List[int]
    R: List[int]
    S: List[List[int]]
    constraints: ClassVar[List[Constraints]] = [Constraints.P, Constraints.R, Constraints.S]
    objective: ClassVar[Objective] = Objective.Cmax

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.constructive


class Heuristics(Methods.Heuristics):
    
    
    @staticmethod
    def constructive(instance: risijCmax_Instance):
        """the greedy constructive heuristic to find an initial solution of risijCmax problem minimalizing the factor of (processing time + setup time) of the job to schedule at a given time

        Args:
            instance (risijCmax_Instance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = SingleMachine.SingleSolution(instance=instance)
        remaining_jobs_list = [i for i in range(instance.n)]
        while len(remaining_jobs_list) != 0:
            min_factor = None
            for i in remaining_jobs_list:
                current_machine_schedule = solution.machine
                if (current_machine_schedule.last_job == -1):
                    startTime = max(current_machine_schedule.objective_value,
                                    instance.R[i])
                    factor = startTime + instance.P[i] + \
                        instance.S[i][i]  # Added Sj_ii for rabadi
                else:
                    startTime = max(current_machine_schedule.objective_value,
                                    instance.R[i])
                    factor = startTime + instance.P[i] + instance.S[
                        current_machine_schedule.last_job][i]

                if not min_factor or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
                    taken_startTime = startTime
            if (solution.machine.last_job == -1):
                ci = taken_startTime + instance.P[taken_job] + \
                    instance.S[taken_job][taken_job]  # Added Sj_ii for rabadi
            else:
                ci = taken_startTime + instance.P[taken_job]+ instance.S[
                        solution.machine.last_job][taken_job]
            solution.machine.objective_value = ci
            solution.machine.last_job = taken_job
            solution.machine.job_schedule.append(
                SingleMachine.Job(taken_job, taken_startTime, min_factor))
            remaining_jobs_list.remove(taken_job)
            if (ci > solution.objective_value):
                solution.objective_value = ci

        return Problem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])


    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

class Metaheuristics(Methods.Metaheuristics):
    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]

