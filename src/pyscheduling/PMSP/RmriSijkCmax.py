import heapq
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from random import randint, uniform
from statistics import mean
from time import perf_counter

import matplotlib.pyplot as plt

import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Constraints, Objective, Solver
import pyscheduling.PMSP.ParallelMachines as ParallelMachines
from pyscheduling.PMSP.ParallelMachines import parallel_instance
import pyscheduling.PMSP.PM_methods as pm_methods
from pyscheduling.Problem import Job

try:
    import gurobipy as gp
except ImportError:
    pass
try:
    from docplex.cp.model import CpoModel
    from docplex.cp.solver.cpo_callback import CpoCallback
except ImportError:
    pass

GUROBI_IMPORTED = True if "gurobipy" in sys.modules else False
DOCPLEX_IMPORTED = True if "docplex" in sys.modules else False

@parallel_instance([Constraints.R, Constraints.S], Objective.Cmax)
class RmriSijkCmax_Instance(ParallelMachines.ParallelInstance):
    
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
        def _csp_transform_solution(msol, X_ij, instance):

            sol = ParallelMachines.ParallelSolution(instance)
            for i in range(instance.m):
                k_tasks = []
                for j in range(instance.n):
                    if len(msol[X_ij[i][j]]) > 0:
                        start = msol[X_ij[i][j]][0]
                        end = msol[X_ij[i][j]][1]
                        k_tasks.append(ParallelMachines.Job(j, start, end))

                k_tasks = sorted(k_tasks, key=lambda x: x[1])
                sol.machines[i].job_schedule = k_tasks

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

                # Compute lower bound
                LB = instance.lower_bound()

                # Computing upper bound
                # init_sol = Heuristics.constructive(instance).best_solution
                # UB = init_sol.objective_value
                UB = 100000

                model = CpoModel("pmspModel")

                trans_matrix = {}
                for k in range(instance.m):
                    k_matrix = [0 for i in range(instance.n + 1)
                                for j in range(instance.n + 1)]
                    for i in range(instance.n):
                        # Setup of the first job
                        k_matrix[i+1] = instance.S[k][i][i]
                        for j in range(instance.n):
                            if i != j:
                                # Setup between i and j
                                k_matrix[(i+1)*(instance.n+1) + j +
                                        1] = instance.S[k][i][j]

                    trans_matrix[k] = k_matrix

                # Cumul function
                usage = model.step_at(0, 0)

                # Mother tasks
                E_j = []
                for i in E:
                    task = model.interval_var(optional=False, name=f'E[{i}]')
                    E_j.append(task)
                    # C-12
                    usage += model.pulse(task, 1)

                # Task execution in machines
                T_ij = {}
                M_ji = {}
                S_ij = {}
                X_ij = {}
                for i in M:
                    for j in E:
                        start_period = (instance.R[j], UB)
                        end_period = (instance.R[j] + instance.P[i][k], UB)

                        # Tasks
                        proc_task = model.interval_var(start=start_period,
                                                    size=instance.P[j][i], optional=True, name=f'T[{i},{j}]')
                        setup_task = model.interval_var(start=start_period,
                                                        optional=True, name=f'S[{i},{j}]')
                        span_task = model.interval_var(start=start_period,
                                                    optional=True, name=f'X[{i},{j}]')

                        # C-13
                        model.add((model.presence_of(proc_task) == model.presence_of(
                            setup_task)).set_name(f'P-PSe[{i},{j}]'))
                        model.add(
                            (model.span(span_task, [setup_task, proc_task])).set_name(f'Sp[{i},{j}]'))
                        model.add((model.end_at_start(setup_task, proc_task)
                                ).set_name(f'De-PS[{i},{j}]'))

                        # Build the arrays
                        T_ij.setdefault(i, []).append(proc_task)
                        S_ij.setdefault(i, []).append(setup_task)
                        X_ij.setdefault(i, []).append(span_task)

                        M_ji.setdefault(j, []).append(span_task)

                # C-15 A task is executed in one machine only
                for j in E:
                    model.add(
                        (model.alternative(E_j[j], M_ji[j], 1)).set_name(f'Alt[{j}]'))

                # A task can only process one task at a time and includes a setup time between consecutive tasks
                Seq_i = {}
                for i in M:
                    # C-14
                    types = list(range(1, instance.n + 1))
                    Seq_i[i] = model.sequence_var(X_ij[i], types=types)
                    model.add(
                        (model.no_overlap(Seq_i[i])).set_name(f'noOver[{i}]'))

                    for j in E:
                        # C-16
                        dep = S_ij[i][j]

                        model.add((model.size_of(dep) ==
                                model.element(trans_matrix[i],
                                                model.type_of_prev(
                                    Seq_i[i], X_ij[i][j], 0, 0) * (instance.n+1)
                            + j + 1)).set_name(f'Dep[{i},{j}]')
                        )

                # C-17
                model.add((usage <= instance.m).set_name(f'usage_fun'))

                # Solve for cmax

                model.add(model.max(model.end_of(E_j[i]) for i in E) >= LB)
                model.add(model.minimize(
                    model.max(model.end_of(E_j[i]) for i in E)))  # Cmax

                mycallback = CSP.MyCallback(stop_times=stop_times)
                model.add_solver_callback(mycallback)

                msol = model.solve(LogVerbosity="Normal", Workers=nb_threads, TimeLimit=time_limit, LogPeriod=1000000,
                                trace_log=False,  add_log_to_solution=True, RelativeOptimalityTolerance=0)

                if log_path:
                    logFile = open(log_path, "w")
                    logFile.write('\n\t'.join(msol.get_solver_log().split("!")))

                sol = CSP._csp_transform_solution(msol, X_ij, instance)

                # Construct the solve result
                kpis = {
                    "ObjBound": msol.get_objective_bounds()[0],
                    "MemUsage": msol.get_infos()["MemoryUsage"]
                }
                prev = -1
                for stop_t in mycallback.stop_times:
                    if stop_t in mycallback.best_values:
                        kpis[f'Obj-{stop_t}'] = mycallback.best_values[stop_t]
                        prev = mycallback.best_values[stop_t]
                    else:
                        kpis[f'Obj-{stop_t}'] = prev

                solve_result = RootProblem.SolveResult(
                    best_solution=sol,
                    runtime=msol.get_infos()["TotalTime"],
                    time_to_best=mycallback.best_sol_time,
                    status=CSP.CPO_STATUS.get(
                        msol.get_solve_status(), RootProblem.SolveStatus.INFEASIBLE),
                    kpis=kpis
                )

                return solve_result

            else:
                print("Docplex import error: you can not use this solver")

if GUROBI_IMPORTED:
    class MILP():

        GUROBI_STATUS = {
            gp.GRB.INFEASIBLE: RootProblem.SolveStatus.INFEASIBLE,
            gp.GRB.OPTIMAL: RootProblem.SolveStatus.OPTIMAL
        }

        @staticmethod
        def format_matrices(instance):
            """Formats the matrices to add the dummy job and fix indices according to mip model

            Args:
                instance (RmSijkCmax_Instance): instance to be solved

            Returns:
                (list[list[list[int]]], list[list[int]]): setup times matrices, processing times matrix
            """
            s_ijk = [[[0 for k in range(instance.n+1)]
                    for j in range(instance.n+1)] for i in range(instance.m)]
            for i in range(instance.m):
                for j in range(1, instance.n+1):
                    for k in range(1, instance.n+1):
                        s_ijk[i][j][k] = instance.S[i][j-1][k-1]

                for k in range(1, instance.n+1):
                    s_ijk[i][0][k] = instance.S[i][k-1][k-1]

            p_ij = [[0 for j in range(instance.n+1)] for i in range(instance.m)]
            for i in range(instance.m):
                for j in range(1, instance.n+1):
                    p_ij[i][j] = instance.P[j-1][i]

            return s_ijk, p_ij

        @staticmethod
        def build_callback(mycallback, stop_times=[300, 600, 3600, 7200]):

            setattr(mycallback, "SOLVE_RESULT", RootProblem.SolveResult())
            setattr(mycallback, "CURR_BEST", None)
            setattr(mycallback, "stop_times", stop_times)
            setattr(mycallback, "best_values", dict())
            setattr(mycallback, "stop_idx", 0)

            return mycallback

        @staticmethod
        def mycallback(model, where):

            if where == gp.GRB.Callback.MIPSOL:
                # MIP solution callback
                time = model.cbGet(gp.GRB.Callback.RUNTIME)
                obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                solcnt = model.cbGet(gp.GRB.Callback.MIPSOL_SOLCNT)

                if not MILP.mycallback.CURR_BEST or MILP.mycallback.CURR_BEST > obj:
                    MILP.mycallback.SOLVE_RESULT.time_to_best = time
                    #MILP.mycallback.SOLVE_RESULT.sol_to_best = solcnt
                    #MILP.mycallback.SOLVE_RESULT.cmax = obj
                    MILP.mycallback.CURR_BEST = obj

                while MILP.mycallback.stop_idx < len(MILP.mycallback.stop_times) and \
                        time > MILP.mycallback.stop_times[MILP.mycallback.stop_idx]:
                    # Go to the next stop time
                    MILP.mycallback.stop_idx += 1

                if MILP.mycallback.stop_idx < len(MILP.mycallback.stop_times):
                    MILP.mycallback.best_values[MILP.mycallback.stop_times[MILP.mycallback.stop_idx]] = obj

        @staticmethod
        def transform_solution(Y_ij, C_j, instance):
            sol = ParallelMachines.ParallelSolution(instance)
            for i in range(instance.m):
                for j in range(1, instance.n+1):
                    if Y_ij[(i, j)].x == 1:  # Job j-1 is scheduled on machine i
                        sol.machines[i].job_schedule.append(
                            ParallelMachines.Job(j-1, -1, C_j[j].x))

            for i in range(instance.m):
                sol.machines[i].job_schedule.sort(key=lambda x: x[2])

            sol.compute_objective()
            return sol

        @staticmethod
        def solve(instance, **kwargs):
            """ Returns the solution using the MILP solver

            Args:
                instance (Instance): Instance object to solve
                log_path (str, optional): Path to the log file to output gurobi log. Defaults to None to disable logging.
                time_limit (int, optional): Time limit for executing the solver. Defaults to 300s.
                threads (int, optional): Number of threads to set for gurobi solver. Defaults to 1.

            Returns:
                SolveResult: The object represeting the solving process result
            """
            if "gurobipy" in sys.modules:
                # Extracting parameters
                log_path = kwargs.get("log_path", None)
                time_limit = kwargs.get("time_limit", 300)
                nb_threads = kwargs.get("threads", 1)
                stop_times = kwargs.get("stop_times", [
                                        time_limit // 4, time_limit // 2, (3*time_limit) // 4, time_limit])

                # Preparing ranges
                M = range(instance.m)
                E = range(instance.n)

                # Compute lower bound
                LB = instance.lower_bound()

                # Computing upper bound
                # init_sol = Heuristics.constructive(instance).best_solution
                # UB = init_sol.cmax

                # Min Ri
                min_ri = min(instance.R)

                model = gp.Model("sched")

                s_ijk, p_ij = MILP.format_matrices(instance)

                N0 = range(0, instance.n+1)
                N = range(1, instance.n+1)
                V = 100000

                X_ijk = model.addVars(instance.m, instance.n+1,
                                    instance.n+1, vtype=gp.GRB.BINARY, name="X")
                Y_ij = model.addVars(instance.m, instance.n+1,
                                    vtype=gp.GRB.BINARY, name="Y")

                C_j = model.addVars(instance.n + 1, lb=0,
                                    vtype=gp.GRB.INTEGER, name="C")
                C_max = model.addVar(lb=0, vtype=gp.GRB.INTEGER, name="C_max")

                # Assignment
                model.addConstrs((sum(Y_ij[i, j]
                                for i in M) == 1 for j in N), name="C2")
                model.addConstrs((Y_ij[i, k] == sum(X_ijk[i, j, k]
                                for j in N0 if j != k) for i in M for k in N), name="C3")
                model.addConstrs((Y_ij[i, j] == sum(X_ijk[i, j, k]
                                for k in N0 if k != j) for i in M for j in N), name="C4")
                model.addConstrs((sum(X_ijk[i, 0, k]
                                for k in N) <= 1 for i in M), name="C5")

                # Scheduling
                model.addConstrs((C_j[k] >= C_j[j] + V * (X_ijk[i, j, k] - 1) + s_ijk[i][j][k] + p_ij[i][k]
                                for j in N0 for k in N for i in M if j != k), name="C6")
                model.addConstrs((C_j[k] >= instance.R[k-1] + V * (X_ijk[i, j, k] - 1) + s_ijk[i][j][k] + p_ij[i][k]
                                for j in N0 for k in N for i in M if j != k), name="C7")
                model.addConstr(C_j[0] == 0, name="C8")
                model.addConstrs((C_j[j] <= C_max for j in N), name="C9")

                # Valid inequalities
                model.addConstrs((sum(s_ijk[i][j][k] * X_ijk[i, j, k] for j in N0 for k in N if j != k) +
                                sum(p_ij[i][j] * Y_ij[i, j] for j in N) + min_ri <= C_max for i in M), name="C10")

                model.setObjective(C_max)

                if time_limit:
                    model.setParam("TimeLimit", time_limit)
                if log_path:
                    model.setParam("LogFile", log_path)

                model.setParam("Threads", nb_threads)
                model.setParam("LogToConsole", 0)
                model.update()
                model.optimize(MILP.build_callback(
                    MILP.mycallback, stop_times=stop_times))

                sol = MILP.transform_solution(Y_ij, C_j, instance)

                # Construct the solve result
                execTimes = {"ObjBound": model.ObjBound}
                prev = -1
                for stop_t in MILP.mycallback.stop_times:
                    if stop_t in MILP.mycallback.best_values:
                        execTimes[f'Obj-{stop_t}'] = MILP.mycallback.best_values[stop_t]
                        prev = MILP.mycallback.best_values[stop_t]
                    else:
                        execTimes[f'Obj-{stop_t}'] = prev

                solve_result = RootProblem.SolveResult(
                    best_solution=sol,
                    runtime=model.Runtime,
                    time_to_best=MILP.mycallback.SOLVE_RESULT.time_to_best,
                    status=MILP.GUROBI_STATUS.get(
                        model.status, RootProblem.SolveStatus.FEASIBLE),
                    kpis=execTimes
                )

                return solve_result
            else:
                print("gurobipy import error: you can not use this solver")


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

    if GUROBI_IMPORTED:
        @staticmethod
        def milp(instance, **kwargs):
            return MILP.solve(instance, **kwargs)
    else:
        @staticmethod
        def milp(instance, **kwargs):
            print("Gurobi import error: you can not use this solver")
            return None

class Heuristics(pm_methods.Heuristics):

    @staticmethod
    def list_heuristic(instance: RmriSijkCmax_Instance, rule: int = 1, decreasing: bool = False):
        """list_heuristic gives the option to use different rules in order to consider given factors in the construction of the solution

        Args:
            instance (RmriSijkCmax_Instance): Instance to be solved by the heuristic
            rule (int, optional): ID of the rule to follow by the heuristic. Defaults to 1.
            decreasing (bool, optional): _description_. Defaults to False.

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        if rule == 1:  # Mean Processings
            remaining_jobs_list = [(i, mean(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 2:  # Min Processings
            remaining_jobs_list = [(i, min(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 3:  # Mean Processings + Mean Setups
            setup_means = [
                mean(means_list)
                for means_list in [[mean(s[i]) for s in instance.S]
                                   for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, mean(instance.P[i]) + setup_means[i])
                                   for i in range(instance.n)]
        elif rule == 4:  # Max Processings
            remaining_jobs_list = [(i, max(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 5:  # IS1
            max_setup = [
                max([max(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, max(max(instance.P[i]), max_setup[i][0]))
                                   for i in range(instance.n)]
        elif rule == 6:  # IS2
            min_setup = [
                min([min(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, max(min(instance.P[i]), min_setup[i][0]))
                                   for i in range(instance.n)]
        elif rule == 7:  # IS3
            min_setup = [
                min([min(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, min(min(instance.P[i]), min_setup[i][0]))
                                   for i in range(instance.n)]
        elif rule == 8:  # IS4
            max_setup = [
                max([max(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, min(max(instance.P[i]), max_setup[i][0]))
                                   for i in range(instance.n)]
        elif rule == 9:  # IS5
            max_setup = [
                max([max(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, max(instance.P[i]) / max_setup[i][0])
                                   for i in range(instance.n)]
        elif rule == 10:  # IS6
            min_setup = [
                min([min(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, min(instance.P[i]) / (min_setup[i][0] + 1))
                                   for i in range(instance.n)]
        elif rule == 11:  # IS7
            max_setup = [
                max([max(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, max_setup[i][0] / max(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 12:  # IS8
            min_setup = [
                min([min(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, min_setup[i][0] / (min(instance.P[i]) + 1))
                                   for i in range(instance.n)]
        elif rule == 13:  # IS9
            min_setup = [
                min([min(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, min_setup[i][0] / max(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 14:  # IS10
            max_setup = [
                max([max(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, max_setup[i][0] / (min(instance.P[i]) + 1))
                                   for i in range(instance.n)]
        elif rule == 15:  # IS11
            max_setup = [
                max([max(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, max_setup[i][0] + max(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 16:  # IS12
            min_setup = [
                min([min(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, min_setup[i][0] + min(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 17:  # IS13
            proc_div_setup = [
                min([instance.P[i][k] / max(instance.S[k][i])]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 18:  # IS14
            proc_div_setup = [
                min(instance.P[i][k] / (min(instance.S[k][i]) + 1)
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 19:  # IS15
            proc_div_setup = [
                max(max(instance.S[k][i]) / instance.P[i][k]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 20:  # IS16
            proc_div_setup = [
                max([min(instance.S[k][i]) / instance.P[i][k]]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 21:  # IS17
            proc_div_setup = [
                min([min(instance.S[k][i]) / instance.P[i][k]]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 22:  # IS18
            proc_div_setup = [
                min([max(instance.S[k][i]) / instance.P[i][k]]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 23:  # IS19
            proc_div_setup = [
                min([max(instance.S[k][i]) + instance.P[i][k]]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 24:  # IS20
            proc_div_setup = [
                max([max(instance.S[k][i]) + instance.P[i][k]]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 25:  # IS21
            proc_div_setup = [
                min(min(instance.S[k][i]) + instance.P[i][k]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 26:  # IS22
            proc_div_setup = [
                max([min(instance.S[k][i]) + instance.P[i][k]]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 27:  # Mean Setup
            setup_means = [
                mean(means_list)
                for means_list in [[mean(s[i]) for s in instance.S]
                                   for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_means[i])
                                   for i in range(instance.n)]
        elif rule == 28:  # Min Setup
            setup_mins = [
                min(min_list) for min_list in [[min(s[i]) for s in instance.S]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_mins[i])
                                   for i in range(instance.n)]
        elif rule == 29:  # Max Setup
            setup_max = [
                max(max_list) for max_list in [[max(s[i]) for s in instance.S]
                                               for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, setup_max[i])
                                   for i in range(instance.n)]
        elif rule == 30:
            remaining_jobs_list = [(i, instance.R[i])
                                   for i in range(instance.n)]
        elif rule == 31:  # Mean Processings + Mean Setups
            setup_means = [
                mean(means_list)
                for means_list in [[mean(s[i]) for s in instance.S]
                                   for i in range(instance.n)]
            ]
            remaining_jobs_list = [(i, mean(instance.P[i]) + setup_means[i] + instance.R[i])
                                   for i in range(instance.n)]
        elif rule == 32:  # IS10
            max_setup = [
                max([max(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, instance.R[i] + max_setup[i][0] / (min(instance.P[i]) + 1))
                                   for i in range(instance.n)]
        elif rule == 33:  # IS21
            proc_div_setup = [
                min(min(instance.S[k][i]) + instance.P[i][k]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, instance.R[i] + proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 34:  # IS2
            min_setup = [
                min([min(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, instance.R[i] + max(min(instance.P[i]), min_setup[i][0]))
                                   for i in range(instance.n)]
        elif rule == 35:  # Min Processings
            remaining_jobs_list = [(i, instance.R[i] + min(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 36:  # IS14
            proc_div_setup = [
                min(instance.P[i][k] / (min(instance.S[k][i]) + 1)
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, instance.R[i] + proc_div_setup[i])
                                   for i in range(instance.n)]
        elif rule == 37:  # IS12
            min_setup = [
                min([min(instance.S[k][i])] for k in range(instance.m))
                for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, instance.R[i] + min_setup[i][0] + min(instance.P[i]))
                                   for i in range(instance.n)]
        elif rule == 38:  # IS15
            proc_div_setup = [
                max(max(instance.S[k][i]) / instance.P[i][k]
                    for k in range(instance.m)) for i in range(instance.n)
            ]
            remaining_jobs_list = [(i, instance.R[i] + proc_div_setup[i])
                                   for i in range(instance.n)]

        remaining_jobs_list = sorted(remaining_jobs_list,
                                     key=lambda job: job[1],
                                     reverse=decreasing)
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
