from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import List

import pyscheduling.FS.FlowShop as FS
from pyscheduling import Problem
from pyscheduling.Problem import Job
from pyscheduling.core.base_solvers import BaseSolver

try:
    import docplex
    from docplex.cp.expression import INTERVAL_MAX
    from docplex.cp.model import CpoModel
    from docplex.cp.solver.cpo_callback import CpoCallback
except ImportError:
    pass

DOCPLEX_IMPORTED = True if "docplex" in sys.modules else False

if DOCPLEX_IMPORTED:

    @dataclass
    class CSP(BaseSolver):
        
        time_limit : int = 300
        log_path : Path = field(default = None)
        nb_threads : int = 1
        stop_times : List[int] = field(default= None)

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
        def _csp_transform_solution(msol, E_i, instance: FS.FlowShopInstance):

            sol = FS.FlowShopSolution(instance)
            for k in range(instance.m):
                k_tasks = []
                for i in range(instance.n):
                    start = msol[E_i[i][k]][0]
                    end = msol[E_i[i][k]][1]
                    k_tasks.append(Job(i,start,end))

                    k_tasks = sorted(k_tasks, key= lambda x: x[1])
                    sol.machines[k].oper_schedule = [job[0] for job in k_tasks]
            
            sol.job_schedule = sol.machines[0].oper_schedule
            
            return sol
        
        def solve(self, instance: FS.FlowShopInstance):
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

            # Extracting parameters
            self.notify_on_start()
            objective = instance.get_objective()
            if self.stop_times is None:
                self.stop_times = [self.time_limit // 4, self.time_limit // 2, (self.time_limit * 3) // 4, self.time_limit]

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
            if objective == Problem.Objective.Cmax:
                model.add( model.minimize( model.max(model.end_of(job_i) for i in E for job_i in E_i[i]) ) )
            elif objective == Problem.Objective.wiCi:
                model.add(model.minimize( sum( instance.W[i] * model.end_of(E_i[i][-1]) for i in E ) )) # sum_{i in E} wi * ci
            elif objective == Problem.Objective.wiFi:
                model.add(model.minimize( sum( instance.W[i] * (model.end_of(E_i[i][-1]) - instance.R[i]) for i in E ) )) # sum_{i in E} wi * (ci - ri)
            elif objective == Problem.Objective.wiTi:
                model.add( model.minimize( 
                    sum( instance.W[i] * model.max(model.end_of(E_i[i]) - instance.D[i], 0) for i in E ) # sum_{i in E} wi * Ti
                ))
            
            # Link the callback to save stats of the solve process
            mycallback = CSP.MyCallback(stop_times=self.stop_times)
            model.add_solver_callback(mycallback)

            # Run the model
            msol = model.solve(LogVerbosity="Normal", Workers=self.nb_threads, TimeLimit=self.time_limit, LogPeriod=1000000,
                            log_output=True, trace_log=False, add_log_to_solution=True, RelativeOptimalityTolerance=0)

            # Logging solver's infos if log_path is specified
            if self.log_path:
                with open(self.log_path, "a") as logFile:
                    logFile.write('\n\t'.join(msol.get_solver_log().split("!")))
                    logFile.flush()

            sol = CSP._csp_transform_solution(msol, E_i, instance)
            self.notify_on_solution_found(sol)

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

            self.notify_on_complete()
            self.solve_result.kpis = kpis

            return self.solve_result
        
else:

    @dataclass
    class CSP(BaseSolver):

        def solve(self, instance: FS.FlowShopInstance):
            print("Docplex import error: you can not use this solver")
            return None
        
            
