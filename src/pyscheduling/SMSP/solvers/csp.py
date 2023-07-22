from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import List

from pyscheduling import Problem
from pyscheduling.core.base_solvers import BaseSolver

try:
    from docplex.cp.expression import INTERVAL_MAX
    from docplex.cp.model import CpoModel
    from docplex.cp.solver.cpo_callback import CpoCallback
except ImportError:
    pass

DOCPLEX_IMPORTED = True if "docplex" in sys.modules else False


if DOCPLEX_IMPORTED:
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

        def _csp_transform_solution(self, msol, E_i, instance):

            sol = instance.create_solution()
            k_tasks = []
            for i in range(instance.n):
                start = msol[E_i[i]][0]
                end = msol[E_i[i]][1]
                k_tasks.append(Problem.Job(i,start,end))
                
                k_tasks = sorted(k_tasks, key= lambda x: x[1])
                sol.machine.job_schedule = k_tasks
            
            sol.compute_objective()

            return sol
        
        def solve(self, instance):
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
            self.notify_on_start()
            objective = instance.get_objective()
            if self.stop_times is None:
                self.stop_times = [self.time_limit // 4, self.time_limit // 2, (self.time_limit * 3) // 4, self.time_limit]
            
            E = range(instance.n)

            # Construct the model
            model = CpoModel("smspModel")

            # Jobs interval_vars including the release date and processing times constraints
            E_i = []
            for i in E:
                start_period = (instance.R[i], INTERVAL_MAX) if hasattr(instance, 'R') else (0, INTERVAL_MAX)
                job_i = model.interval_var( start = start_period,
                                            size = instance.P[i], optional= False, name=f'E[{i}]')
                E_i.append(job_i)

            # Sequential execution on the machine
            machine_sequence = model.sequence_var( E_i, list(E) )
            model.add( model.no_overlap(machine_sequence) )
            
            # Define the objective 
            if objective == Problem.Objective.wiCi:
                model.add(model.minimize( sum( instance.W[i] * model.end_of(E_i[i]) for i in E ) )) # sum_{i in E} wi * ci
            elif objective == Problem.Objective.wiTi:
                model.add( model.minimize( 
                    sum( instance.W[i] * model.max(model.end_of(E_i[i]) - instance.D[i], 0) for i in E ) # sum_{i in E} wi * Ti
                ))
            elif objective == Problem.Objective.Cmax:
                model.add(model.minimize( max( model.end_of(E_i[i]) for i in E ) )) # max_{i in E} ci 

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