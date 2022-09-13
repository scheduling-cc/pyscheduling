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
from pyscheduling.Problem import Solver
import pyscheduling.PMSP.ParallelMachines as ParallelMachines
import pyscheduling.PMSP.PM_methods as pm_methods

try:
    import docplex
    from docplex.cp.model import CpoModel
    from docplex.cp.solver.cpo_callback import CpoCallback
except ImportError:
    pass

DOCPLEX_IMPORTED = True if "docplex" in sys.modules else False

@dataclass
class RmSijkCmax_Instance(ParallelMachines.ParallelInstance):
    P: list[list[int]] = field(default_factory=list)  # Processing time
    S: list[list[list[int]]] = field(default_factory=list)  # Setup time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            RmSijkCmax_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of configuration
        m = int(ligne0[2])  # number of jobs
        i = 2
        instance = cls("test", n, m)
        instance.P, i = instance.read_P(content, i)
        instance.S, i = instance.read_S(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int, configuration_number: int, protocol: ParallelMachines.GenerationProtocol = ParallelMachines.GenerationProtocol.VALLADA, law: ParallelMachines.GenerationLaw = ParallelMachines.GenerationLaw.UNIFORM, Pmin: int = -1, Pmax: int = -1, Gamma: float = 0.0, Smin:  int = -1, Smax: int = -1, InstanceName: str = ""):
        """Random generation of RmSijkCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            configuration_number (int): number of machines of the instance
            protocol (ParallelMachines.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to ParallelMachines.GenerationProtocol.VALLADA.
            law (ParallelMachines.GenerationLaw, optional): probablistic law of generation. Defaults to ParallelMachines.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            Gamma (float, optional): Setup time factor. Defaults to 0.0.
            Smin (int, optional): Minimal setup time. Defaults to -1.
            Smax (int, optional): Maximal setup time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            RmSijkCmax_Instance: the randomly generated instance
        """
        if(Pmin == -1):
            Pmin = randint(1, 100)
        if(Pmax == -1):
            Pmax = randint(Pmin, 100)
        if(Gamma == 0.0):
            Gamma = round(uniform(1.0, 3.0), 1)
        if(Smin == -1):
            Smin = randint(1, 100)
        if(Smax == -1):
            Smax = randint(Smin, 100)
        instance = cls(InstanceName, jobs_number, configuration_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.S = instance.generate_S(
            protocol, law, instance.P, Gamma, Smin, Smax)
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+"  "+str(self.m)+"\n")
        f.write(str(self.m)+"\n")
        for i in range(self.n):
            for j in range(self.m):
                f.write("\t"+str(j)+"\t"+str(self.P[i][j]))
            f.write("\n")
        f.write("SSD\n")
        for i in range(self.m):
            f.write("M"+str(i)+"\n")
            for j in range(self.n):
                for k in range(self.n):
                    f.write(str(self.S[i][j][k])+"\t")
                f.write("\n")
        f.close()

    def init_sol_method(self):
        return Heuristics.constructive

    def get_objective(self):
        return RootProblem.Objective.Cmax
    
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
                for i in E:  # (t for t in E if t != j ):
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
            
            sol.cmax()
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

class Heuristics():

    @staticmethod
    def constructive(instance: RmSijkCmax_Instance):
        """the greedy constructive heuristic to find an initial solution of RmSijkCmax problem minimalizing the factor of (processing time + setup time) of the job to schedule at a given time

        Args:
            instance (RmSijkCmax_Instance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = ParallelMachines.ParallelSolution(instance=instance)
        remaining_jobs_list = [i for i in range(instance.n)]
        while len(remaining_jobs_list) != 0:
            min_factor = None
            for i in remaining_jobs_list:
                for j in range(instance.m):
                    current_machine_schedule = solution.machines[j]
                    if (current_machine_schedule.last_job == -1):
                        factor = current_machine_schedule.completion_time + \
                            instance.P[i][j]
                    else:
                        factor = current_machine_schedule.completion_time + \
                            instance.P[i][j] + \
                            instance.S[j][current_machine_schedule.last_job][i]

                    if not min_factor or (min_factor > factor):
                        min_factor = factor
                        taken_job = i
                        taken_machine = j
            if (solution.machines[taken_machine].last_job == -1):
                ci = solution.machines[taken_machine].completion_time + \
                    instance.P[taken_job][taken_machine]
            else:
                ci = solution.machines[taken_machine].completion_time + instance.P[taken_job][taken_machine] + \
                    instance.S[taken_machine][solution.machines[taken_machine].last_job][taken_job]

            solution.machines[taken_machine].job_schedule.append(ParallelMachines.Job(
                taken_job, solution.machines[taken_machine].completion_time, ci))
            solution.machines[taken_machine].completion_time = ci
            solution.machines[taken_machine].last_job = taken_job

            remaining_jobs_list.remove(taken_job)
            if (ci > solution.objective_value):
                solution.objective_value = ci

        return RootProblem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

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
        for element in remaining_jobs_list:
            i = element[0]
            min_factor = None
            for j in range(instance.m):
                current_machine_schedule = solution.machines[j]
                if (current_machine_schedule.last_job == -1):  # First Job
                    factor = current_machine_schedule.completion_time + \
                        instance.P[i][j]
                else:
                    factor = current_machine_schedule.completion_time + \
                        instance.P[i][j] + \
                        instance.S[j][current_machine_schedule.last_job][i]

                if not min_factor or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
                    taken_machine = j

            if (solution.machines[taken_machine].last_job == -1):
                ci = solution.machines[taken_machine].completion_time + \
                    instance.P[taken_job][taken_machine]
            else:
                ci = solution.machines[taken_machine].completion_time + instance.P[taken_job][taken_machine] + \
                    instance.S[taken_machine][solution.machines[taken_machine].last_job][taken_job]
            solution.machines[taken_machine].job_schedule.append(ParallelMachines.Job(
                taken_job, solution.machines[taken_machine].completion_time, ci))
            solution.machines[taken_machine].completion_time = ci
            solution.machines[taken_machine].last_job = taken_job
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


class Metaheuristics(pm_methods.Metaheuristics_Cmax):
    @staticmethod
    def antColony(instance: RmSijkCmax_Instance, **data):
        """Returns the solution using the ant colony algorithm

        Args:
            instance (RmSijkCmax_Instance): Instance to be solved

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = RootProblem.SolveResult()
        AC = AntColony(instance=instance, **data)
        solveResult.best_solution, solveResult.all_solutions = AC.solve()
        solveResult.solve_status = RootProblem.SolveStatus.FEASIBLE
        solveResult.runtime = perf_counter() - startTime
        return solveResult


class AntColony(object):

    def __init__(self, instance: RmSijkCmax_Instance, n_ants: int = 60, n_best: int = 1,
                 n_iterations: int = 100, alpha=1, beta=1, phi: float = 0.081, evaporation: float = 0.01,
                 q0: float = 0.5, best_ants: int = 10, pheromone_init=10):
        """
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
        all_time_shortest_cmax = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_solutions = self.gen_all_paths()
            all_solutions = self.improve_best_ants(all_solutions)

            shortest_path = min(all_solutions, key=lambda x: x[1])
            longest_path = max(all_solutions, key=lambda x: x[1])

            if shortest_path[1] == longest_path[1]:
                self.reinit_graph()

            if shortest_path[1] < all_time_shortest_cmax[1]:
                all_time_shortest_cmax = shortest_path
            self.spread_pheronome_global(all_solutions)

        return all_time_shortest_cmax[0], [solution[0] for solution in all_solutions]

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

    def spread_pheronome_global(self, all_solutions: list[ParallelMachines.ParallelSolution]):
        """Update pheromone levels globally after finding new solutions

        Args:
            all_solutions (list[RmSijkCmax_Solution]): list of generated solutions
        """
        sorted_solutions = sorted(all_solutions, key=lambda x: x[1])

        for solution, cmax_i in sorted_solutions[:self.n_best]:
            for k in range(solution.instance.m):
                machine_k = solution.machines[k]
                for i, task_i in enumerate(machine_k.job_schedule):
                    self.aco_graph[0][task_i.id,
                                      k] += self.phi * self.LB / cmax_i
                    if i > 0:
                        prev_task = machine_k.job_schedule[i-1]
                        self.aco_graph[1][k, prev_task.id,
                                          task_i.id] += self.phi * self.LB / cmax_i

    def improve_best_ants(self, all_solutions):
        """Apply local search to the best solution

        Args:
            all_solutions (_type_): list of all generated solutions

        Returns:
            list[RmSijkCmax_Solution]: list of updated solutions
        """
        sorted_solutions = sorted(all_solutions, key=lambda x: x[1])
        local_search = ParallelMachines.PM_LocalSearch()
        for solution, cmax_i in sorted_solutions[:self.best_ants]:
            solution = local_search.improve(solution)
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
                all_solutions.append((solution_i, solution_i.objective_value))

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
        solution_path = ParallelMachines.ParallelSolution(self.instance)

        for m in range(len(affectation)):
            machine_schedule = []
            if len(affectation[m]) > 0:
                first_task = affectation[m][random.randrange(
                    0, len(affectation[m]))]
                machine_schedule.append(ParallelMachines.Job(first_task, 0, 0))
                prev = first_task

                for i in range(len(affectation[m]) - 1):
                    pheromone_i = pheromone[m, prev]
                    next_task = self.pick_task(prev, m, pheromone_i, affectation[m], [
                                               job.id for job in machine_schedule])

                    # Spread Pheromone Locally
                    pheromone_i[next_task] = (
                        1 - self.evaporation) * pheromone_i[next_task]

                    machine_schedule.append(
                        ParallelMachines.Job(next_task, 0, 0))

            current_machine = solution_path.machines[m]
            current_machine.job_schedule = machine_schedule
            current_machine.compute_completion_time(self.instance)

        solution_path.cmax()

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
