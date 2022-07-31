import heapq
import random
import sys
from dataclasses import dataclass, field
from math import exp
from pathlib import Path
from random import randint, uniform
from statistics import mean
from time import perf_counter

import matplotlib.pyplot as plt

import pyscheduling_cc.ParallelMachines as ParallelMachines
import pyscheduling_cc.Problem as Problem
from pyscheduling_cc.Problem import Solver

try:
    import docplex
    import gurobipy as gp
    from docplex.cp.model import CpoModel
    from docplex.cp.solver.cpo_callback import CpoCallback
except ImportError:
    pass


@dataclass
class RmriSijkCmax_Instance(ParallelMachines.ParallelInstance):
    P: list[list[int]] = field(default_factory=list)  # Processing time
    S: list[list[list[int]]] = field(default_factory=list)  # Setup time
    R: list[int] = field(default_factory=list)  # Release time

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
        instance.R, i = instance.read_R(content, i)
        instance.S, i = instance.read_S(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int, configuration_number: int, protocol: ParallelMachines.GenerationProtocol = ParallelMachines.GenerationProtocol.VALLADA, law: ParallelMachines.GenerationLaw = ParallelMachines.GenerationLaw.UNIFORM, Pmin: int = -1, Pmax: int = -1, Alpha: float = 0.0, Gamma: float = 0.0, Smin:  int = -1, Smax: int = -1, InstanceName: str = ""):
        """Random generation of RmSijkCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            configuration_number (int): number of machines of the instance
            protocol (ParallelMachines.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to ParallelMachines.GenerationProtocol.VALLADA.
            law (ParallelMachines.GenerationLaw, optional): probablistic law of generation. Defaults to ParallelMachines.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            Alpha (float,optional): Release time factor. Defaults to 0.0.
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
        if(Alpha == 0.0):
            Alpha = round(uniform(1.0, 3.0), 1)
        if(Gamma == 0.0):
            Gamma = round(uniform(1.0, 3.0), 1)
        if(Smin == -1):
            Smin = randint(1, 100)
        if(Smax == -1):
            Smax = randint(Smin, 100)
        instance = cls(InstanceName, jobs_number, configuration_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.R = instance.generate_R(
            protocol, law, instance.P, Pmin, Pmax, Alpha)
        instance.S = instance.generate_S(
            protocol, law, instance.P, Gamma, Smin, Smax)
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+" "+str(self.m)+"\n")
        f.write(str(self.m)+"\n")
        for i in range(self.n):
            for j in range(self.m):
                f.write(str(self.P[i][j])+"\t")
            f.write("\n")
        f.write("Release time\n")
        for i in range(self.n):
            f.write(str(self.R[i])+"\t")
        f.write("\n")
        f.write("SSD\n")
        for i in range(self.m):
            f.write("M"+str(i)+"\n")
            for j in range(self.n):
                for k in range(self.n):
                    f.write(str(self.S[i][j][k])+"\t")
                f.write("\n")
        f.close()

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


@dataclass
class RmriSijkCmax_Solution(ParallelMachines.ParallelSolution):

    def __init__(self, instance: RmriSijkCmax_Instance = None, configuration: list[ParallelMachines.Machine] = None, objective_value: int = 0):
        """Constructor of RmSijkCmax_Solution
        Args:
            instance (RmriSijkCmax_Instance, optional): Instance to be solved by the solution. Defaults to None.
            configuration (list[ParallelMachines.Machine], optional): list of machines of the instance. Defaults to None.
            objective_value (int, optional): initial objective value of the solution. Defaults to 0.
        """
        self.instance = instance
        if configuration is None:
            self.configuration = []
            for i in range(instance.m):
                machine = ParallelMachines.Machine(i, 0, -1, [])
                self.configuration.append(machine)
        else:
            self.configuration = configuration
        self.objective_value = 0

    def __str__(self):
        return "Cmax : " + str(self.objective_value) + "\n" + "Machine_ID | Job_schedule (job_id , start_time , completion_time) | Completion_time\n" + "\n".join(map(str, self.configuration))

    def __eq__(self, other):
        for i, machine in enumerate(self.configuration):
            if (machine != other.configuration[i]):
                return False

        return True

    def __lt__(self, other):
        return self.objective_value < other.objective_value

    def copy(self):
        copy_machines = []
        for m in self.configuration:
            copy_machines.append(m.copy())

        copy_solution = RmriSijkCmax_Solution(self.instance)
        for i in range(self.instance.m):
            copy_solution.configuration[i] = copy_machines[i]
        copy_solution.objective_value = self.objective_value
        return copy_solution

    @classmethod
    def read_txt(cls, path: Path):
        """Read a solution from a txt file

        Args:
            path (Path): path to the solution's txt file of type Path from pathlib

        Returns:
            RmSijkCmax_Solution:
        """
        f = open(path, "r")
        content = f.read().split('\n')
        objective_value_ = int(content[0].split(':')[1])
        configuration_ = []
        for i in range(2, len(content)):
            line_content = content[i].split('|')
            configuration_.append(ParallelMachines.Machine(int(line_content[0]), int(line_content[2]), job_schedule=[ParallelMachines.Job(
                int(j[0]), int(j[1]), int(j[2])) for j in [job.strip()[1:len(job.strip())-1].split(',') for job in line_content[1].split(':')]]))
        solution = cls(objective_value=objective_value_,
                       configuration=configuration_)
        return solution

    def plot(self, path: Path = None) -> None:
        """Plot the solution in an appropriate diagram"""
        if "matplotlib" in sys.modules:
            if self.instance is not None:
                # Add Tasks ID
                fig, gnt = plt.subplots()

                # Setting labels for x-axis and y-axis
                gnt.set_xlabel('seconds')
                gnt.set_ylabel('Machines')

                # Setting ticks on y-axis

                ticks = []
                ticks_labels = []
                for i in range(len(self.configuration)):
                    ticks.append(10*(i+1) + 5)
                    ticks_labels.append(str(i+1))

                gnt.set_yticks(ticks)
                # Labelling tickes of y-axis
                gnt.set_yticklabels(ticks_labels)

                # Setting graph attribute
                gnt.grid(True)

                for j in range(len(self.configuration)):
                    schedule = self.configuration[j].job_schedule
                    prev = -1
                    prevEndTime = 0
                    for element in schedule:
                        job_index, startTime, endTime = element
                        if prevEndTime < startTime:
                            # Idle Time
                            gnt.broken_barh(
                                [(prevEndTime, startTime - prevEndTime)], ((j+1) * 10, 9), facecolors=('tab:gray'))
                        if prev != -1:
                            # Setup Time
                            gnt.broken_barh([(startTime, self.instance.S[j][prev][job_index])], ((
                                j+1) * 10, 9), facecolors=('tab:orange'))
                            # Processing Time
                            gnt.broken_barh([(startTime + self.instance.S[j][prev][job_index],
                                            self.instance.P[job_index][j])], ((j+1) * 10, 9), facecolors=('tab:blue'))
                        else:
                            gnt.broken_barh([(startTime, self.instance.P[job_index][j])], ((
                                j+1) * 10, 9), facecolors=('tab:blue'))
                        prev = job_index
                        prevEndTime = endTime
                if path:
                    plt.savefig(path)
                else:
                    plt.show()
                return
            else:
                print("Please assign the solved instance to the solution object")
        else:
            print("Matplotlib is not installed, you can't use gant_plot")
            return

    def is_valid(self):
        """
        Check if solution respects the constraints
        """
        set_jobs = set()
        is_valid = True
        for machine in self.configuration:
            prev_job = None
            ci, setup_time, expected_start_time = 0, 0, 0
            for i, element in enumerate(machine.job_schedule):
                job, startTime, endTime = element
                # Test End Time + start Time
                if prev_job is None:
                    setup_time = self.instance.S[machine.machine_num][job][job]
                    expected_start_time = self.instance.R[job]
                else:
                    setup_time = self.instance.S[machine.machine_num][prev_job][job]
                    expected_start_time = max(ci, self.instance.R[job])

                proc_time = self.instance.P[job][machine.machine_num]
                ci = expected_start_time + proc_time + setup_time

                if startTime != expected_start_time or endTime != ci:
                    print(f'## Error: in machine {machine.machine_num}' +
                          f' found {element} expected {job,expected_start_time, ci}')
                    is_valid = False
                set_jobs.add(job)
                prev_job = job

        is_valid &= len(set_jobs) == self.instance.n
        return is_valid


class ExactSolvers():

    @staticmethod
    def csp(instance, **kwargs):
        return CSP.solve(instance, **kwargs)

    @staticmethod
    def milp(instance, **kwargs):
        return MILP.solve(instance, **kwargs)


class CSP():

    CPO_STATUS = {
        "Feasible": Problem.SolveStatus.FEASIBLE,
        "Optimal": Problem.SolveStatus.OPTIMAL
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

        sol = RmriSijkCmax_Solution(instance)
        for i in range(instance.m):
            k_tasks = []
            for j in range(instance.n):
                if len(msol[X_ij[i][j]]) > 0:
                    start = msol[X_ij[i][j]][0]
                    end = msol[X_ij[i][j]][1]
                    k_tasks.append(ParallelMachines.Job(j, start, end))

            k_tasks = sorted(k_tasks, key=lambda x: x[1])
            sol.configuration[i].job_schedule = k_tasks

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

            solve_result = Problem.SolveResult(
                best_solution=sol,
                runtime=msol.get_infos()["TotalTime"],
                time_to_best=mycallback.best_sol_time,
                status=CSP.CPO_STATUS.get(
                    msol.get_solve_status(), Problem.SolveStatus.INFEASIBLE),
                kpis=kpis
            )

            return solve_result

        else:
            print("Docplex import error: you can not use this solver")


class MILP():

    GUROBI_STATUS = {
        gp.GRB.INFEASIBLE: Problem.SolveStatus.INFEASIBLE,
        gp.GRB.OPTIMAL: Problem.SolveStatus.OPTIMAL
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

        setattr(mycallback, "SOLVE_RESULT", Problem.SolveResult())
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
        sol = RmriSijkCmax_Solution(instance)
        for i in range(instance.m):
            for j in range(1, instance.n+1):
                if Y_ij[(i, j)].x == 1:  # Job j-1 is scheduled on machine i
                    sol.configuration[i].job_schedule.append(
                        ParallelMachines.Job(j-1, -1, C_j[j].x))

        for i in range(instance.m):
            sol.configuration[i].job_schedule.sort(key=lambda x: x[2])

        sol.cmax()
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

            solve_result = Problem.SolveResult(
                best_solution=sol,
                runtime=model.Runtime,
                time_to_best=MILP.mycallback.SOLVE_RESULT.time_to_best,
                status=MILP.GUROBI_STATUS.get(
                    model.status, Problem.SolveStatus.FEASIBLE),
                kpis=execTimes
            )

            return solve_result
        else:
            print("gurobipy import error: you can not use this solver")


class Heuristics():

    @staticmethod
    def constructive(instance: RmriSijkCmax_Instance):
        """the greedy constructive heuristic to find an initial solution of RmSijkCmax problem minimalizing the factor of (processing time + setup time) of the job to schedule at a given time

        Args:
            instance (RmriSijkCmax_Instance): Instance to be solved by the heuristic


        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = RmriSijkCmax_Solution(instance=instance)

        remaining_jobs_list = [j for j in range(instance.n)]

        while len(remaining_jobs_list) != 0:
            min_factor = None
            for i in remaining_jobs_list:
                for j in range(instance.m):
                    current_machine_schedule = solution.configuration[j]
                    if (current_machine_schedule.last_job == -1):
                        startTime = max(current_machine_schedule.completion_time,
                                        instance.R[i])
                        factor = startTime + instance.P[i][j] + \
                            instance.S[j][i][i]  # Added Sj_ii for rabadi
                    else:
                        startTime = max(current_machine_schedule.completion_time,
                                        instance.R[i])
                        factor = startTime + instance.P[i][j] + instance.S[j][
                            current_machine_schedule.last_job][i]

                    if not min_factor or (min_factor > factor):
                        min_factor = factor
                        taken_job = i
                        taken_machine = j
                        taken_startTime = startTime
            if (solution.configuration[taken_machine].last_job == -1):
                ci = taken_startTime + instance.P[taken_job][taken_machine] + \
                    instance.S[taken_machine][taken_job][taken_job]  # Added Sj_ii for rabadi
            else:
                ci = taken_startTime + instance.P[taken_job][
                    taken_machine] + instance.S[taken_machine][
                        solution.configuration[taken_machine].last_job][taken_job]
            solution.configuration[taken_machine].completion_time = ci
            solution.configuration[taken_machine].last_job = taken_job
            solution.configuration[taken_machine].job_schedule.append(
                ParallelMachines.Job(taken_job, taken_startTime, min_factor))
            remaining_jobs_list.remove(taken_job)
            if (ci > solution.objective_value):
                solution.objective_value = ci

        return Problem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

    @staticmethod
    def ordered_constructive(instance: RmriSijkCmax_Instance, remaining_jobs_list=None, is_random: bool = False):
        """the ordered greedy constructive heuristic to find an initial solution of RmSijkCmax problem minimalizing the factor of (processing time + setup time) of
        jobs in the given order on different machines

        Args:
            instance (RmriSijkCmax_Instance): Instance to be solved by the heuristic
            remaining_jobs_list (list[int],optional): specific job sequence to consider by the heuristic
            is_random (bool,optional): shuffle the remaining_jobs_list if it's generated by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        start_time = perf_counter()
        solution = RmriSijkCmax_Solution(instance)
        if remaining_jobs_list is None:
            remaining_jobs_list = [i for i in range(instance.n)]
            if is_random:
                random.shuffle(remaining_jobs_list)

        for i in remaining_jobs_list:
            min_factor = None
            for j in range(instance.m):
                current_machine_schedule = solution.configuration[j]
                if (current_machine_schedule.last_job == -1):
                    startTime = max(current_machine_schedule.completion_time,
                                    instance.R[i])
                    factor = startTime + instance.P[i][j] + \
                        instance.S[j][i][i]  # Added Sj_ii for rabadi
                else:
                    startTime = max(current_machine_schedule.completion_time,
                                    instance.R[i])
                    factor = startTime + instance.P[i][j] + instance.S[j][
                        current_machine_schedule.last_job][i]

                if not min_factor or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
                    taken_machine = j
                    taken_startTime = startTime

            # Apply the move
            if (current_machine_schedule.last_job == -1):
                ci = taken_startTime + instance.P[taken_job][taken_machine] +\
                    instance.S[taken_machine][taken_job][taken_job]  # Added Sj_ii for rabadi
            else:
                ci = taken_startTime + instance.P[taken_job][
                    taken_machine] + instance.S[taken_machine][
                        solution.configuration[taken_machine].last_job][taken_job]
            solution.configuration[taken_machine].completion_time = ci
            solution.configuration[taken_machine].last_job = taken_job
            solution.configuration[taken_machine].job_schedule.append(
                ParallelMachines.Job(taken_job, taken_startTime, min_factor))
            if (ci > solution.objective_value):
                solution.objective_value = ci

        return Problem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

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

    @staticmethod
    def meta_raps(instance: RmriSijkCmax_Instance, p: float, r: int, nb_exec: int):
        """Returns the solution using the meta-raps algorithm

        Args:
            instance (RmriSijkCmax_Instance): The instance to be solved by the metaheuristic
            p (float): probability of taking the greedy best solution
            r (int): percentage of moves to consider to select the best move
            nb_exec (int): Number of execution of the metaheuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = Problem.SolveResult()
        solveResult.all_solutions = []
        best_solution = None
        for _ in range(nb_exec):
            solution = RmriSijkCmax_Solution(instance)
            remaining_jobs_list = [i for i in range(instance.n)]
            toDelete = 0
            while len(remaining_jobs_list) != 0:
                toDelete += 1
                insertions_list = []
                for i in remaining_jobs_list:
                    for j in range(instance.m):
                        current_machine_schedule = solution.configuration[j]
                        insertions_list.append(
                            (i, j, 0, current_machine_schedule.completion_time_insert(i, 0, instance)))
                        for k in range(1, len(current_machine_schedule.job_schedule)):
                            insertions_list.append(
                                (i, j, k, current_machine_schedule.completion_time_insert(i, k, instance)))

                insertions_list = sorted(
                    insertions_list, key=lambda insertion: insertion[3])
                proba = random.random()
                if proba < p:
                    rand_insertion = insertions_list[0]
                else:
                    rand_insertion = random.choice(
                        insertions_list[0:int(instance.n * r)])
                taken_job, taken_machine, taken_pos, ci = rand_insertion
                solution.configuration[taken_machine].job_schedule.insert(
                    taken_pos, ParallelMachines.Job(taken_job, 0, 0))
                solution.configuration[taken_machine].compute_completion_time(
                    instance, taken_pos)
                if taken_pos == len(solution.configuration[taken_machine].job_schedule)-1:
                    solution.configuration[taken_machine].last_job = taken_job
                if ci > solution.objective_value:
                    solution.objective_value = ci
                remaining_jobs_list.remove(taken_job)

            solution.fix_cmax()
            solveResult.all_solutions.append(solution)
            if not best_solution or best_solution.objective_value > solution.objective_value:
                best_solution = solution

        solveResult.best_solution = best_solution
        solveResult.runtime = perf_counter() - startTime
        solveResult.solve_status = Problem.SolveStatus.FEASIBLE
        return solveResult

    @staticmethod
    def grasp(instance: RmriSijkCmax_Instance, x: float, nb_exec: int):
        """Returns the solution using the grasp algorithm

        Args:
            instance (RmSijkCmax_Instance): Instance to be solved by the metaheuristic
            x (float): percentage of moves to consider to select the best move
            nb_exec (int): Number of execution of the metaheuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = Problem.SolveResult()
        solveResult.all_solutions = []
        best_solution = None
        for _ in range(nb_exec):
            solution = RmriSijkCmax_Solution(instance)
            remaining_jobs_list = [i for i in range(instance.n)]
            while len(remaining_jobs_list) != 0:
                insertions_list = []
                for i in remaining_jobs_list:
                    for j in range(instance.m):
                        current_machine_schedule = solution.configuration[j]
                        insertions_list.append(
                            (i, j, 0, current_machine_schedule.completion_time_insert(i, 0, instance)))
                        for k in range(1, len(current_machine_schedule.job_schedule)):
                            insertions_list.append(
                                (i, j, k, current_machine_schedule.completion_time_insert(i, k, instance)))

                insertions_list = sorted(
                    insertions_list, key=lambda insertion: insertion[3])
                rand_insertion = random.choice(
                    insertions_list[0:int(instance.n * x)])
                taken_job, taken_machine, taken_pos, ci = rand_insertion
                solution.configuration[taken_machine].job_schedule.insert(
                    taken_pos, ParallelMachines.Job(taken_job, 0, 0))
                solution.configuration[taken_machine].compute_completion_time(
                    instance, taken_pos)
                if taken_pos == len(solution.configuration[taken_machine].job_schedule)-1:
                    solution.configuration[taken_machine].last_job = taken_job
                remaining_jobs_list.remove(taken_job)

            solution.fix_cmax()
            solveResult.all_solutions.append(solution)
            if not best_solution or best_solution.objective_value > solution.objective_value:
                best_solution = solution

        solveResult.best_solution = best_solution
        solveResult.runtime = perf_counter() - startTime
        solveResult.solve_status = Problem.SolveStatus.FEASIBLE
        return solveResult

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]


class Metaheuristics():
    @staticmethod
    def lahc(instance: RmriSijkCmax_Instance, **kwargs):
        """ Returns the solution using the LAHC algorithm
        Args:
            instance (RmSijkCmax_Instance): Instance object to solve
            Lfa (int, optional): Size of the candidates list. Defaults to 25.
            Nb_iter (int, optional): Number of iterations of LAHC. Defaults to 300.
            Non_improv (int, optional): LAHC stops when the number of iterations without
                improvement is achieved. Defaults to 50.
            LS (bool, optional): Flag to apply local search at each iteration or not.
                Defaults to True.
            time_limit_factor: Fixes a time limit as follows: n*m*time_limit_factor if specified, 
                else Nb_iter is taken Defaults to None
            init_sol_method: The method used to get the initial solution. 
                Defaults to "constructive"
            seed (int, optional): Seed for the random operators to make the algo deterministic
        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """

        # Extracting parameters
        time_limit_factor = kwargs.get("time_limit_factor", None)
        init_sol_method = kwargs.get(
            "init_sol_method", Heuristics.constructive)
        Lfa = kwargs.get("Lfa", 30)
        Nb_iter = kwargs.get("Nb_iter", 500000)
        Non_improv = kwargs.get("Non_improv", 50000)
        LS = kwargs.get("LS", True)
        seed = kwargs.get("seed", None)

        if seed:
            random.seed(seed)

        first_time = perf_counter()
        if time_limit_factor:
            time_limit = instance.m * instance.n * time_limit_factor

        # Generate init solutoin using the initial solution method
        solution_init = init_sol_method(instance).best_solution

        if not solution_init:
            return Problem.SolveResult()

        local_search = ParallelMachines.PM_LocalSearch()

        if LS:
            solution_init = local_search.improve(
                solution_init)  # Improve it with LS

        all_solutions = []
        solution_best = solution_init.copy()  # Save the current best solution
        all_solutions.append(solution_best)
        lahc_list = [solution_init.objective_value] * Lfa  # Create LAHC list

        N = 0
        i = 0
        time_to_best = perf_counter() - first_time
        current_solution = solution_init
        while i < Nb_iter and N < Non_improv:
            # check time limit if exists
            if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                break

            solution_i = ParallelMachines.NeighbourhoodGeneration.lahc_neighbour(
                current_solution)

            if LS:
                solution_i = local_search.improve(solution_i)
            if solution_i.objective_value < current_solution.objective_value or solution_i.objective_value < lahc_list[i % Lfa]:

                current_solution = solution_i
                if solution_i.objective_value < solution_best.objective_value:
                    all_solutions.append(solution_i)
                    solution_best = solution_i
                    time_to_best = (perf_counter() - first_time)
                    N = 0
            lahc_list[i % Lfa] = solution_i.objective_value
            i += 1
            N += 1

        # Construct the solve result
        solve_result = Problem.SolveResult(
            best_solution=solution_best,
            solutions=all_solutions,
            runtime=(perf_counter() - first_time),
            time_to_best=time_to_best,
        )

        return solve_result

    @staticmethod
    def SA(instance: RmriSijkCmax_Instance, **kwargs):
        """ Returns the solution using the simulated annealing algorithm or the restricted simulated annealing
        algorithm
        Args:
            instance (RmSijkCmax_Instance): Instance object to solve
            T0 (float, optional): Initial temperature. Defaults to 1.1.
            Tf (float, optional): Final temperature. Defaults to 0.01.
            k (float, optional): Acceptance facture. Defaults to 0.1.
            b (float, optional): Cooling factor. Defaults to 0.97.
            q0 (int, optional): Probability to apply restricted swap compared to
            restricted insertion. Defaults to 0.5.
            n_iter (int, optional): Number of iterations for each temperature. Defaults to 10.
            Non_improv (int, optional): SA stops when the number of iterations without
                improvement is achieved. Defaults to 500.
            LS (bool, optional): Flag to apply local search at each iteration or not. 
                Defaults to True.
            time_limit_factor: Fixes a time limit as follows: n*m*time_limit_factor if specified, 
                else Nb_iter is taken Defaults to None
            init_sol_method: The method used to get the initial solution. 
                Defaults to "constructive"
            seed (int, optional): Seed for the random operators to make the 
                algo deterministic if fixed. Defaults to None.

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """

        # Extracting the parameters
        restriced = kwargs.get("restricted", False)
        time_limit_factor = kwargs.get("time_limit_factor", None)
        init_sol_method = kwargs.get(
            "init_sol_method", Heuristics.constructive)
        T0 = kwargs.get("T0", 1.4)
        Tf = kwargs.get("Tf", 0.01)
        k = kwargs.get("k", 0.1)
        b = kwargs.get("b", 0.99)
        q0 = kwargs.get("q0", 0.5)
        n_iter = kwargs.get("n_iter", 20)
        Non_improv = kwargs.get("Non_improv", 5000)
        LS = kwargs.get("LS", True)
        seed = kwargs.get("seed", None)

        if restriced:
            generationMethod = ParallelMachines.NeighbourhoodGeneration.RSA_neighbour
            data = {'q0': q0}
        else:
            generationMethod = ParallelMachines.NeighbourhoodGeneration.SA_neighbour
            data = {}
        if seed:
            random.seed(seed)

        first_time = perf_counter()
        if time_limit_factor:
            time_limit = instance.m * instance.n * time_limit_factor

        solution_init = init_sol_method(instance).best_solution

        if not solution_init:
            return Problem.SolveResult()

        local_search = ParallelMachines.PM_LocalSearch()

        if LS:
            solution_init = local_search.improve(solution_init)

        all_solutions = []
        # Initialisation
        T = T0
        N = 0
        time_to_best = 0
        solution_i = None
        all_solutions.append(solution_init)
        solution_best = solution_init
        while T > Tf and (N != Non_improv):
            # check time limit if exists
            if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                break
            for i in range(0, n_iter):
                # check time limit if exists
                if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                    break

                # solution_i = ParallelMachines.NeighbourhoodGeneration.generate_NX(solution_best)  # Generate solution in Neighbour
                solution_i = generationMethod(solution_best, **data)
                if LS:
                    # Improve generated solution using LS
                    solution_i = local_search.improve(solution_i)

                delta_cmax = solution_init.objective_value - solution_i.objective_value
                if delta_cmax >= 0:
                    solution_init = solution_i
                else:
                    r = random.random()
                    factor = delta_cmax / (k * T)
                    exponent = exp(factor)
                    if (r < exponent):
                        solution_init = solution_i

                if solution_best.objective_value > solution_init.objective_value:
                    all_solutions.append(solution_init)
                    solution_best = solution_init
                    time_to_best = (perf_counter() - first_time)
                    N = 0

            T = T * b
            N += 1

        # Construct the solve result
        solve_result = Problem.SolveResult(
            best_solution=solution_best,
            runtime=(perf_counter() - first_time),
            time_to_best=time_to_best,
            solutions=all_solutions
        )

        return solve_result

    @staticmethod
    def GA(instance: RmriSijkCmax_Instance, **kwargs):
        """Returns the solution using the genetic algorithm

        Args:
            instance (RmSijkCmax_Instance): Instance to be solved

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = Problem.SolveResult()
        solveResult.best_solution, solveResult.all_solutions = GeneticAlgorithm.solve(
            instance, **kwargs)
        solveResult.solve_status = Problem.SolveStatus.FEASIBLE
        solveResult.runtime = perf_counter() - startTime
        return solveResult

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]


class GeneticAlgorithm():

    @staticmethod
    def solve(instance: RmriSijkCmax_Instance, pop_size=50, p_cross=0.7, p_mut=0.5, p_ls=1, pressure=30, nb_iter=100):
        population = GeneticAlgorithm.generate_population(
            instance, pop_size, LS=(p_ls != 0))
        delta = 0
        i = 0
        N = 0
        local_search = ParallelMachines.PM_LocalSearch()
        best_cmax = None
        best_solution = None
        solutions = []
        while i < nb_iter and N < 20:  # ( instance.n*instance.m*50/2000 ):
            # Select the parents
            # print("Selection")
            parent_1, parent_2 = GeneticAlgorithm.selection(
                population, pressure)

            # Cross parents
            pc = random.uniform(0, 1)
            if pc < p_cross:
                # print("Crossover")
                child1, child2 = GeneticAlgorithm.crossover(
                    instance, parent_1, parent_2)
            else:
                child1, child2 = parent_1, parent_2

            # Mutation
            # Child 1
            pm = random.uniform(0, 1)
            if pm < p_mut:
                #print("mutation 1 ")
                child1 = GeneticAlgorithm.mutation(instance, child1)
            # Child 2
            pm = random.uniform(0, 1)
            if pm < p_mut:
                #print("Mutation 2")
                child2 = GeneticAlgorithm.mutation(instance, child2)

            # Local Search
            # Child 1
            pls = random.uniform(0, 1)
            if pls < p_ls:
                #print("LS 1")
                child1 = local_search.improve(child1)
            # Child 2
            pls = random.uniform(0, 1)
            if pls < p_ls:
                #print("LS 2")
                child2 = local_search.improve(child2)

            best_child = child1 if child1.objective_value <= child2.objective_value else child2
            if not best_cmax or best_child.objective_value < best_cmax:
                solutions.append(best_child)
                best_cmax = best_child.objective_value
                best_solution = best_child
                N = 0
            # Replacement
            # print("Remplacement")
            GeneticAlgorithm.replacement(population, child1)
            GeneticAlgorithm.replacement(population, child2)
            i += 1
            N += 1

        return best_solution, solutions

    @staticmethod
    def generate_population(instance: RmriSijkCmax_Instance, pop_size=40, LS=True):
        population = []
        i = 0
        nb_solution = 0
        nb_rules = 38
        heapq.heapify(population)
        local_search = ParallelMachines.PM_LocalSearch()
        while nb_solution < pop_size:
            # Generate a solution using a heuristic
            if i == 0:
                solution_i = Heuristics.constructive(instance).best_solution
            elif i <= nb_rules:
                solution_i = Heuristics.list_heuristic(
                    instance, **{"rule": i}).best_solution
            elif i <= nb_rules * 2:
                solution_i = Heuristics.list_heuristic(
                    instance, **{"rule": i - nb_rules, "decreasing": True}).best_solution
            else:
                solution_i = Heuristics.ordered_constructive(
                    instance, **{"is_random": True}).best_solution

            if LS:
                solution_i = local_search.improve(solution_i)

            if solution_i not in population:
                #print(i, solution_i.cmax)
                # population.append(solution_i)
                heapq.heappush(population, solution_i)
                nb_solution += 1
            i += 1

        return population

    @staticmethod
    def selection(population, pressure):
        #parents_list = random.sample(population,pressure)
        #best_index = min(enumerate(parents_list),key=lambda x: x[1].cmax)[0]
        parent1 = population[0]

        parent2 = random.choice(population)
        #best_index = min(enumerate(parents_list),key=lambda x: x[1].cmax)[0]
        #parent2 = parents_list[best_index]

        return parent1, parent2

    @staticmethod
    def crossover(instance: RmriSijkCmax_Instance, parent_1, parent_2):
        child1 = RmriSijkCmax_Solution(instance)
        child2 = RmriSijkCmax_Solution(instance)

        for i, machine1 in enumerate(parent_1.configuration):
            machine2 = parent_2.configuration[i]
            # generate 2 random crossing points
            cross_point_1 = random.randint(0, len(machine1.job_schedule)-1)
            cross_point_2 = random.randint(0, len(machine2.job_schedule)-1)

            child1.configuration[i].job_schedule.extend(
                machine1.job_schedule[0:cross_point_1])
            child2.configuration[i].job_schedule.extend(
                machine2.job_schedule[0:cross_point_2])

            child1.configuration[i].completion_time = child1.configuration[i].compute_completion_time(
                instance)
            child2.configuration[i].completion_time = child2.configuration[i].compute_completion_time(
                instance)

            GeneticAlgorithm.complete_solution(instance, parent_1, child2)
            GeneticAlgorithm.complete_solution(instance, parent_2, child1)

        return child1, child2

    @staticmethod
    def mutation(instance: RmriSijkCmax_Instance, child: RmriSijkCmax_Solution):
        # Random Internal Swap
        child = ParallelMachines.NeighbourhoodGeneration.random_swap(
            child, force_improve=False, internal=True)
        return child

    @staticmethod
    def complete_solution(instance: RmriSijkCmax_Instance, parent, child: RmriSijkCmax_Solution):
        # Cache the jobs affected to both childs
        child_jobs = set(
            job[0] for machine in child.configuration for job in machine.job_schedule)
        for i, machine_parent in enumerate(parent.configuration):
            for job in machine_parent.job_schedule:
                if job[0] not in child_jobs:
                    child = ParallelMachines.PM_LocalSearch.best_insertion_machine(
                        child, i, job[0])

        child.fix_cmax()

    @staticmethod
    def replacement(population, child):
        if child not in population:
            worst_index = max(enumerate(population),
                              key=lambda x: x[1].objective_value)[0]
            if child.objective_value <= population[worst_index].objective_value:
                heapq.heappushpop(population, child)
                #population[worst_index] = child
                return True

        return False
