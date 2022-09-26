import random
import sys
from time import perf_counter

import pyscheduling.Problem as RootProblem
import pyscheduling.FS.FlowShop as FlowShop

try:
    import docplex
    from docplex.cp.expression import INTERVAL_MAX
    from docplex.cp.model import CpoModel
    from docplex.cp.solver.cpo_callback import CpoCallback
except ImportError:
    pass

DOCPLEX_IMPORTED = True if "docplex" in sys.modules else False


class Heuristics_Cmax():

    def MINIT(instance : FlowShop.FlowShopInstance):
        """Gupta's MINIT heuristic which is based on iteratively scheduling a new job at the end
        so that it minimizes the idle time at the last machine

        Args:
            instance (FlowShop.FlowShopInstance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        start_time = perf_counter()
        solution = FlowShop.FlowShopSolution(instance=instance)

        #step 1 : Find pairs of jobs (job_i,job_j) which minimizes the idle time
        min_idleTime = None
        idleTime_ij_list = []
        for job_i in range(instance.n):
            for job_j in range(instance.n):
                if job_i != job_j :
                    pair = (job_i,job_j)
                    solution.job_schedule = [job_i,job_j] 
                    solution.cmax()
                    idleTime_ij = solution.idle_time()
                    idleTime_ij_list.append((pair,idleTime_ij))
                    if min_idleTime is None or idleTime_ij<min_idleTime : min_idleTime = idleTime_ij

        min_IT_list = [pair_idleTime_couple for pair_idleTime_couple in idleTime_ij_list if pair_idleTime_couple[1] == min_idleTime]
        #step 2 : Break the tie by choosing the pair based on performance at increasingly earlier machines (m-2,m-3,..)
        # For simplicity purposes, a random choice is performed
        min_IT = random.choice(min_IT_list)
                
        taken_pair = min_IT[0]
        job_schedule = [taken_pair[0],taken_pair[1]]
        solution.job_schedule = job_schedule
        solution.cmax()
        #step 3 :
        remaining_jobs_list = [job_id for job_id in list(range(instance.n)) if job_id not in job_schedule]

        while len(remaining_jobs_list) > 0 :
            min_IT_factor = None
            old_idleTime = solution.idle_time()
            for job_id in remaining_jobs_list:
                last_job_startTime, new_cmax = solution.idle_time_cmax_insert_last_pos(job_id)
                factor = old_idleTime + (last_job_startTime - solution.machines[instance.m-1].objective)
                if min_IT_factor is None or factor < min_IT_factor:
                    min_IT_factor = factor
                    taken_job = job_id
            job_schedule.append(taken_job)
            remaining_jobs_list.remove(taken_job)
            solution.job_schedule = job_schedule
            solution.cmax(len(job_schedule)-1)


        return RootProblem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

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
        def _csp_transform_solution(msol, E_i, instance: FlowShop.FlowShopInstance):

            sol = FlowShop.FlowShopSolution(instance)
            for k in range(instance.m):
                k_tasks = []
                for i in range(instance.n):
                    start = msol[E_i[i][k]][0]
                    end = msol[E_i[i][k]][1]
                    k_tasks.append(FlowShop.Job(i,start,end))

                    k_tasks = sorted(k_tasks, key= lambda x: x[1])
                    sol.machines[k].job_schedule = [job[0] for job in k_tasks]
            
            sol.job_schedule = sol.machines[0].job_schedule
            
            return sol
        
        @staticmethod
        def solve(instance, **kwargs):
            """ Returns the solution using the Cplex - CP optimizer solver

            Args:
                instance (Instance): Instance object to solve
                objective (str): The objective to optimize. Defaults to wiCi
                log_path (str, optional): Path to the log file to output cp optimizer log. Defaults to None to disable logging.
                time_limit (int, optional): Time limit for executing the solver. Defaults to 300s.
                threads (int, optional): Number of threads to set for cp optimizer solver. Defaults to 1.

            Returns:
                SolveResult: The object represeting the solving process result
            """
            if "docplex" in sys.modules:
                # Extracting parameters
                objective = kwargs.get("objective", "wiCi")
                log_path = kwargs.get("log_path", None)
                time_limit = kwargs.get("time_limit", 300)
                nb_threads = kwargs.get("threads", 1)
                stop_times = kwargs.get(
                    "stop_times", [time_limit // 4, time_limit // 2, (time_limit * 3) // 4, time_limit])

                E = range(instance.n)
                M = range(instance.m)

                model = CpoModel("FS_Model")

                # Preparing transition matrices
                trans_matrix = {}
                if hasattr(instance, 'S'):
                    for k in range(instance.m):
                        k_matrix = [ [0 for _ in range(instance.n + 1)] for _ in range(instance.n + 1) ]
                        for i in range(instance.n):
                            ele = instance.S[k][i][i]
                            k_matrix[i+1][0] = ele
                            k_matrix[0][i+1] = ele

                            for j in range(instance.n):
                                k_matrix[i+1][j+1] = instance.S[k][i][j]

                        trans_matrix[k] = model.transition_matrix(k_matrix)
                    
                    # Create a dummy job for the first task
                    first_task = model.interval_var(size=0, optional= False, start = 0, name=f'first_task')

                E_i = [[] for i in E]
                M_k = [[] for k in M]
                types_k = [ list(range(1, instance.n + 1)) for k in M ]
                for i in E:
                    for k in M:
                        start_period = (instance.R[i], INTERVAL_MAX) if hasattr(instance, 'R') else (0, INTERVAL_MAX)
                        job_i = model.interval_var( start = start_period,
                                                    size = instance.P[i][k], optional= False, name=f'E[{i},{k}]')
                        E_i[i].append(job_i)
                        M_k[k].append(job_i)

                # No overlap inside machines
                seq_array = []
                for k in M:
                    if hasattr(instance, 'S'):
                        seq_k = model.sequence_var([first_task] + M_k[k], [0] + types_k[k], name=f"Seq_{k}")
                        model.add( model.no_overlap(seq_k, trans_matrix[k]) )
                    else:
                        seq_k = model.sequence_var(M_k[k], types_k[k], name=f"Seq_{k}")
                        model.add( model.no_overlap(seq_k) )
                        
                    seq_array.append(seq_k)
                    
                # Same sequence constraint
                for k in range(1, instance.m):
                    model.add( model.same_sequence(seq_array[k - 1], seq_array[k]) )

                # Precedence constraint between machines for each job
                for i in E:
                    for k in range(1, instance.m):
                        model.add( model.end_before_start(E_i[i][k - 1], E_i[i][k]) )

                # Add objective
                model.add( model.minimize( model.max(model.end_of(job_i) for i in E for job_i in E_i[i]) ) )

                # Link the callback to save stats of the solve process
                mycallback = CSP.MyCallback(stop_times=stop_times)
                model.add_solver_callback(mycallback)

                # Run the model
                msol = model.solve(LogVerbosity="Normal", Workers=nb_threads, TimeLimit=time_limit, LogPeriod=1000000,
                                log_output=True, trace_log=False, add_log_to_solution=True, RelativeOptimalityTolerance=0)

                # Logging solver's infos if log_path is specified
                if log_path:
                    with open(log_path, "a") as logFile:
                        logFile.write('\n\t'.join(msol.get_solver_log().split("!")))
                        logFile.flush()

                sol = CSP._csp_transform_solution(msol, E_i, instance)

                # Construct the solve result
                kpis = {
                    "ObjValue": msol.get_objective_value(),
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
