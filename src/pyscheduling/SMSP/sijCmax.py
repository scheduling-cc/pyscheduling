from time import perf_counter

import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Constraints, Objective
from pyscheduling.SMSP.SingleMachine import single_instance
import pyscheduling.SMSP.SingleMachine as SingleMachine
import pyscheduling.SMSP.SM_Methods as Methods
from pyscheduling.SMSP.SM_Methods import ExactSolvers


@single_instance([Constraints.S], Objective.Cmax)
class sijCmax_Instance(SingleMachine.SingleInstance):

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.constructive



class Heuristics(Methods.Heuristics):
    
    
    @staticmethod
    def constructive(instance: sijCmax_Instance):
        """the greedy constructive heuristic to find an initial solution of sijCmax problem minimalizing the factor of (processing time + setup time) of the job to schedule at a given time

        Args:
            instance (sijCmax_Instance): Instance to be solved by the heuristic

        Returns:
            RootProblem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = SingleMachine.SingleSolution(instance=instance)
        remaining_jobs_list = [i for i in range(instance.n)]
        while len(remaining_jobs_list) != 0:
            min_factor = None
            for i in remaining_jobs_list:

                current_machine_schedule = solution.machine
                if (current_machine_schedule.last_job == -1):
                    factor = current_machine_schedule.objective_value + \
                        instance.P[i]
                else:
                    factor = current_machine_schedule.objective_value + \
                        instance.P[i] + \
                        instance.S[current_machine_schedule.last_job][i]

                if not min_factor or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
            if (solution.machine.last_job == -1):
                ci = solution.machine.objective_value + \
                    instance.P[taken_job]
            else:
                ci = solution.machine.objective_value + instance.P[taken_job]+ \
                    instance.S[solution.machine.last_job][taken_job]

            solution.machine.job_schedule.append(SingleMachine.Job(
                taken_job, solution.machine.objective_value, ci))
            solution.machine.objective_value = ci
            solution.machine.last_job = taken_job

            remaining_jobs_list.remove(taken_job)
            if (ci > solution.objective_value):
                solution.objective_value = ci

        return RootProblem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])


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

