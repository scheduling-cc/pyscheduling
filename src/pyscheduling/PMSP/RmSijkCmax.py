import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from random import randint, uniform
from statistics import mean
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice as np_choice

import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Constraints, Objective, Solver
import pyscheduling.PMSP.ParallelMachines as ParallelMachines
from pyscheduling.PMSP.ParallelMachines import parallel_instance
import pyscheduling.PMSP.PM_methods as pm_methods
from pyscheduling.Problem import Job

try:
    import docplex
    from docplex.cp.model import CpoModel
    from docplex.cp.solver.cpo_callback import CpoCallback
except ImportError:
    pass

DOCPLEX_IMPORTED = True if "docplex" in sys.modules else False

@parallel_instance([Constraints.S], Objective.Cmax)
class RmSijkCmax_Instance(ParallelMachines.ParallelInstance):

    def init_sol_method(self):
        return Heuristics.BIBA
    
    def lower_bound(self):
        """Computes the lower bound of maximal completion time of the instance 
        by dividing the sum of minimal completion time between job pairs on the number of machines

        Returns:
            int: Lower Bound of maximal completion time
        """
        # Preparing ranges
        M = range(self.m)
        E = range(self.n)
        # Compute lower bound
        sum_j = 0
        all_max_r_j = 0
        for j in E:
            min_j = None
            min_r_j = None
            for k in M:
                for i in E:  # (i for i in E if i != j ):
                    if min_j is None or self.P[j][k] + self.S[k][i][j] < min_j:
                        min_j = self.P[j][k] + self.S[k][i][j]
                    if min_r_j is None or self.P[j][k] + self.S[k][i][j] < min_r_j:
                        min_r_j = self.P[j][k] + self.S[k][i][j]
            sum_j += min_j
            all_max_r_j = max(all_max_r_j, min_r_j)

        lb1 = sum_j / self.m
        LB = max(lb1, all_max_r_j)

        return LB

if DOCPLEX_IMPORTED:
    class CSP():

        CPO_STATUS = {
            "Feasible": RootProblem.SolveStatus.FEASIBLE,
            "Optimal": RootProblem.SolveStatus.OPTIMAL
        }

        class MyCallback(CpoCallback):
            """A callback used to log the value of cmax at different timestamps

            Args:
                CpoCallback (_type_): Inherits from CpoCallback
            """

            def __init__(self, stop_times=[300, 600, 3600, 7200]):
                self.stop_times = stop_times
                self.best_values = dict()
                self.stop_idx = 0
                self.best_sol_time = 0
                self.nb_sol = 0

            def invoke(self, solver, event, jsol):

                if event == "Solution":
                    self.nb_sol += 1
                    solve_time = jsol.get_info('SolveTime')
                    self.best_sol_time = solve_time

                    # Go to the next stop time
                    while self.stop_idx < len(self.stop_times) and solve_time > self.stop_times[self.stop_idx]:
                        self.stop_idx += 1

                    if self.stop_idx < len(self.stop_times):
                        # Get important elements
                        obj_val = jsol.get_objective_values()[0]
                        self.best_values[self.stop_times[self.stop_idx]] = obj_val

        @staticmethod
        def _transform_csp_solution(msol, T_ki, instance):
            """Transforms cp optimizer interval variable into a solution 

            Args:
                msol (): CPO solution
                T_ki (list[list[interval_var]]): Interval variables represening jobs
                instance (RmSijkCmax_Instance): instance corresponding to the solution

            Returns:
                ParallelSolution: cpoptimizer's solution
            """
            sol = ParallelMachines.ParallelSolution(instance)
            for k in range(instance.m):
                k_tasks = []
                for i in range(instance.n):
                    if len(msol[T_ki[k][i]]) > 0:
                        start = msol[T_ki[k][i]][0]
                        end = msol[T_ki[k][i]][1]
                        k_tasks.append(ParallelMachines.Job(i, start, end))
                
                k_tasks = sorted(k_tasks, key= lambda x: x[1])
                sol.machines[k].job_schedule = k_tasks
            
            sol.compute_objective(sol.instance)
            return sol

        @staticmethod
        def solve(instance, **kwargs):
            """ Returns the solution using the Cplex - CP optimizer solver

            Args:
                instance (Instance): Instance object to solve
                log_path (str, optional): Path to the log file to output cp optimizer log. Defaults to None to disable logging.
                time_limit (int, optional): Time limit for executing the solver. Defaults to 300s.
                threads (int, optional): Number of threads to set for cp optimizer solver. Defaults to 1.

            Returns:
                SolveResult: The object represeting the solving process result
            """
            if "docplex" in sys.modules:
                
                # Extracting parameters
                log_path = kwargs.get("log_path", None)
                time_limit = kwargs.get("time_limit", 300)
                nb_threads = kwargs.get("threads", 1)
                stop_times = kwargs.get(
                    "stop_times", [time_limit // 4, time_limit // 2, (time_limit * 3) // 4, time_limit])

                # Preparing ranges
                M = range(instance.m)
                E = range(instance.n)

                LB = instance.lower_bound()
                model = CpoModel("pmspModel")

                # Preparing transition matrices
                trans_matrix = {}
                for k in range(instance.m):
                    k_matrix = [ [0 for _ in range(instance.n + 1)] for _ in range(instance.n + 1) ]
                    for i in range(instance.n):
                        ele = instance.S[k][i][i]
                        k_matrix[i+1][0] = ele
                        k_matrix[0][i+1] = ele
                        
                        for j in range(instance.n):
                            k_matrix[i+1][j+1] = instance.S[k][i][j]
                    
                    trans_matrix[k] = model.transition_matrix(k_matrix)

                # Cumul function
                usage = model.step_at(0,0)

                # Mother tasks
                E_j = []
                for i in E:
                    task = model.interval_var( optional= False, name=f'E[{i}]')
                    E_j.append(task)
                    usage += model.pulse(task,1)

                T_ki = {}
                M_ik = {}
                for k in M:
                    for i in E:
                        task = model.interval_var( size=instance.P[i][k], optional= True, name=f'T[{k},{i}]')
                        T_ki.setdefault(k,[]).append(task)
                        M_ik.setdefault(i,[]).append(task)

                # A task is executed in one machine only
                for i in E:
                    model.add( model.alternative(E_j[i], M_ik[i], 1) )

                # A task can only process one task at a time and includes a setup time between consecutive tasks
                first_task = model.interval_var(size=0, optional= False, name=f'first_task')
                model.add( model.start_of(first_task) == 0 )
                Seq_k = {}
                for k in M:
                    types = list(range(instance.n + 1))
                    seq_list = [first_task]
                    seq_list.extend(T_ki[k])
                    Seq_k[k] = model.sequence_var( seq_list, types )
                    model.add( model.no_overlap(Seq_k[k], trans_matrix[k]) ) 
                
                # Cumulative function usage limit
                model.add(usage <= instance.m)

                # Lower bound and objective
                model.add( model.max( model.end_of(E_j[i]) for i in E ) >= LB ) 
                model.add(model.minimize( model.max( model.end_of(E_j[i]) for i in E ))) # Cmax

                # Adding callback to log stats
                mycallback = CSP.MyCallback(stop_times=stop_times)
                model.add_solver_callback(mycallback)

                msol = model.solve(LogVerbosity="Normal", Workers=nb_threads, TimeLimit=time_limit, LogPeriod=1000000,
                                trace_log=False,  add_log_to_solution=True, RelativeOptimalityTolerance=0)

                if log_path:
                    logFile = open(log_path, "w")
                    logFile.write('\n\t'.join(msol.get_solver_log().split("!")))

                sol = CSP._transform_csp_solution(msol, T_ki, instance)

                # Construct the solve result
                kpis = {
                    "ObjBound": msol.get_objective_bounds()[0],
                    "MemUsage": msol.get_infos()["MemoryUsage"]
                }
                
                solve_result = RootProblem.SolveResult(
                    best_solution=sol,
                    runtime=msol.get_infos()["TotalTime"],
                    time_to_best= mycallback.best_sol_time,
                    status=CSP.CPO_STATUS.get(
                        msol.get_solve_status(), RootProblem.SolveStatus.INFEASIBLE),
                    kpis=kpis
                )

                return solve_result

            else:
                print("Docplex import error: you can not use this solver")

class ExactSolvers():

    if DOCPLEX_IMPORTED:
        @staticmethod
        def csp(instance, **kwargs):
            return CSP.solve(instance, **kwargs)
    else:
        @staticmethod
        def csp(instance, **kwargs):
            print("Docplex import error: you can not use this solver")
            return None

class Heuristics(pm_methods.Heuristics):

    @staticmethod
    def list_heuristic(instance: RmSijkCmax_Instance, rule=1, decreasing=False):
        """list_heuristic gives the option to use different rules in order to consider given factors in the construction of the solution

        Args:
            instance (_type_): Instance to be solved by the heuristic
            rule (int, optional): ID of the rule to follow by the heuristic. Defaults to 1.
            decreasing (bool, optional): _description_. Defaults to False.

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = ParallelMachines.ParallelSolution(instance=instance)
        if rule == 1:  # Mean Processings
            remaining_jobs_list = [(i, mean(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 2:  # Min Processings
            remaining_jobs_list = [(i, min(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 3:  # Mean Processings + Mean Setups
            setup_means = [mean(means_list) for means_list in [
                [mean(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [
                (i, mean(instance.P[i])+setup_means[i]) for i in range(instance.n)]
        elif rule == 4:  # Max Processings
            remaining_jobs_list = [(i, max(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 5:  # IS1
            max_setup = [max([max(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, max(max(instance.P[i]), max_setup[i][0])) for i in range(instance.n)]
        elif rule == 6:  # IS2
            min_setup = [min([min(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, max(min(instance.P[i]), min_setup[i][0])) for i in range(instance.n)]
        elif rule == 7:  # IS3
            min_setup = [min([min(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, min(min(instance.P[i]), min_setup[i][0])) for i in range(instance.n)]
        elif rule == 8:  # IS4
            max_setup = [max([max(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, min(max(instance.P[i]), max_setup[i][0])) for i in range(instance.n)]
        elif rule == 9:  # IS5
            max_setup = [max([max(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, max(instance.P[i])/max_setup[i][0]) for i in range(instance.n)]
        elif rule == 10:  # IS6
            min_setup = [min([min(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, min(instance.P[i])/(min_setup[i][0]+1)) for i in range(instance.n)]
        elif rule == 11:  # IS7
            max_setup = [max([max(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, max_setup[i][0]/max(instance.P[i])) for i in range(instance.n)]
        elif rule == 12:  # IS8
            min_setup = [min([min(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, min_setup[i][0]/(min(instance.P[i])+1)) for i in range(instance.n)]
        elif rule == 13:  # IS9
            min_setup = [min([min(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, min_setup[i][0]/max(instance.P[i])) for i in range(instance.n)]
        elif rule == 14:  # IS10
            max_setup = [max([max(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, max_setup[i][0]/(min(instance.P[i])+1)) for i in range(instance.n)]
        elif rule == 15:  # IS11
            max_setup = [max([max(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, max_setup[i][0] + max(instance.P[i])) for i in range(instance.n)]
        elif rule == 16:  # IS12
            min_setup = [min([min(instance.S[k][i])]
                             for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [
                (i, min_setup[i][0] + min(instance.P[i])) for i in range(instance.n)]
        elif rule == 17:  # IS13
            proc_div_setup = [min([instance.P[i][k]/max(instance.S[k][i])]
                                  for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 18:  # IS14
            proc_div_setup = [min([instance.P[i][k]/(min(instance.S[k][i])+1)]
                                  for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 19:  # IS15
            proc_div_setup = [max([max(instance.S[k][i])/instance.P[i][k]]
                                  for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 20:  # IS16
            proc_div_setup = [max([min(instance.S[k][i])/instance.P[i][k]]
                                  for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 21:  # IS17
            proc_div_setup = [min([min(instance.S[k][i])/instance.P[i][k]]
                                  for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 22:  # IS18
            proc_div_setup = [min([max(instance.S[k][i])/instance.P[i][k]]
                                  for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 23:  # IS19
            proc_div_setup = [min([max(instance.S[k][i]) + instance.P[i][k]]
                                  for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 24:  # IS20
            proc_div_setup = [max([max(instance.S[k][i]) + instance.P[i][k]]
                                  for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 25:  # IS21
            proc_div_setup = [min([min(instance.S[k][i]) + instance.P[i][k]]
                                  for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 26:  # IS22
            proc_div_setup = [max([min(instance.S[k][i]) + instance.P[i][k]]
                                  for k in range(instance.m)) for i in range(instance.n)]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 27:  # Mean Setup
            setup_means = [mean(means_list) for means_list in [
                [mean(s[i]) for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [(i, setup_means[i])
                                   for i in range(instance.n)]
        elif rule == 28:  # Min Setup
            setup_mins = [min(min_list) for min_list in [[min(s[i])
                                                          for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [(i, setup_mins[i])
                                   for i in range(instance.n)]
        elif rule == 29:  # Max Setup
            setup_max = [max(max_list) for max_list in [[max(s[i])
                                                         for s in instance.S] for i in range(instance.n)]]
            remaining_jobs_list = [(i, setup_max[i])
                                   for i in range(instance.n)]

        remaining_jobs_list = sorted(
            remaining_jobs_list, key=lambda job: job[1], reverse=decreasing)
        jobs_list = [element[0] for element in remaining_jobs_list]

        return Heuristics.ordered_constructive(instance, remaining_jobs_list=jobs_list)

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]


class Metaheuristics(pm_methods.Metaheuristics):
    pass
