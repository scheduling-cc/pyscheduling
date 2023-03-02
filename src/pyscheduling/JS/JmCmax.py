import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import List

import pyscheduling.JS.JobShop as JobShop
import pyscheduling.JS.JS_methods as js_methods
import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Solver

try:
    import docplex
    from docplex.cp.expression import INTERVAL_MAX
    from docplex.cp.model import CpoModel
    from docplex.cp.solver.cpo_callback import CpoCallback
except ImportError:
    pass

DOCPLEX_IMPORTED = True if "docplex" in sys.modules else False

@dataclass
class JmCmax_Instance(JobShop.JobShopInstance):
    P: List[List[int]] = field(default_factory=list)  # Processing time

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            JmCmax_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of configuration
        m = int(ligne0[1])  # number of jobs
        i = 1
        instance = cls("test", n, m)
        instance.P, i = instance.read_P(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int, configuration_number: int, protocol: JobShop.GenerationProtocol = JobShop.GenerationProtocol.BASE, law: JobShop.GenerationLaw = JobShop.GenerationLaw.UNIFORM, Pmin: int = 10, Pmax: int = 100, InstanceName: str = ""):
        """Random generation of JmCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            configuration_number (int): number of machines of the instance
            protocol (JobShop.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to JobShop.GenerationProtocol.VALLADA.
            law (JobShop.GenerationLaw, optional): probablistic law of generation. Defaults to JobShop.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            JmCmax_Instance: the randomly generated instance
        """
        instance = cls(InstanceName, jobs_number, configuration_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n)+" "+str(self.m)+"\n")
        for job in self.P:
            for operation in job:
                f.write(str(operation[0])+"\t"+str(operation[1])+"\t")
            f.write("\n")
        f.close()

    def init_sol_method(self):
        """Returns the default solving method

        Returns:
            object: default solving method
        """
        return Heuristics.shifting_bottleneck

    def get_objective(self):
        """to get the objective tackled by the instance

        Returns:
            RootProblem.Objective: Makespan
        """
        return RootProblem.Objective.Cmax

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
        def _transform_csp_solution(msol, M_k, types_k, instance):
            """Transforms cp optimizer interval variable into a solution 

            Args:
                msol (): CPO solution
                M_k (list[list[interval_var]]): Interval variables represening jobs inside machines k
                types_k (list[int]): List of job ids inside each machine k
                instance (RmSijkCmax_Instance): instance corresponding to the solution

            Returns:
                JobShopSolution: cpoptimizer's solution
            """
            sol = JobShop.JobShopSolution(instance)
            for k in range(instance.m):
                k_tasks = []
                for i, oper_k_i in enumerate(M_k[k]):
                    start, end, _ = msol[oper_k_i]
                    k_tasks.append(JobShop.Job(types_k[k][i],start,end))

                    k_tasks = sorted(k_tasks, key= lambda x: x[1])
                    sol.machines[k].job_schedule = k_tasks
                
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
                E = range(instance.n)
                M = range(instance.m)

                model = CpoModel("JS_Model")

                E_i = [[] for i in E]
                M_k = [[] for k in M]

                types_k = [ [] for k in M ]
                for i in E:
                    for oper in instance.P[i]:
                        k, p_ik = oper
                        start_period = (0, INTERVAL_MAX)
                        oper_i_k = model.interval_var( start = start_period,
                                                    size = p_ik, optional= False, name=f'E[{i},{k}]')
                        E_i[i].append(oper_i_k)
                        
                        M_k[k].append(oper_i_k)
                        types_k[k].append(i)

                # No overlap inside machines
                seq_array = []
                for k in M:
                    seq_k = model.sequence_var(M_k[k], types_k[k], name=f"Seq_{k}")
                    model.add( model.no_overlap(seq_k) )        
                    seq_array.append(seq_k)

                # Precedence constraint between machines for each job
                for i in E:
                    for k in range(1, len(E_i[i])):
                        model.add( model.end_before_start(E_i[i][k - 1], E_i[i][k]) )

                # Add objective
                model.add( model.minimize( model.max(model.end_of(job_i) for i in E for job_i in E_i[i]) ) )

                # Adding callback to log stats
                mycallback = CSP.MyCallback(stop_times=stop_times)
                model.add_solver_callback(mycallback)

                msol = model.solve(LogVerbosity="Normal", Workers=nb_threads, TimeLimit=time_limit, LogPeriod=1000000,
                                trace_log=False,  add_log_to_solution=True, RelativeOptimalityTolerance=0)

                if log_path:
                    logFile = open(log_path, "w")
                    logFile.write('\n\t'.join(msol.get_solver_log().split("!")))

                sol = CSP._transform_csp_solution(msol, M_k, types_k, instance)

                # Construct the solve result
                kpis = {
                    "ObjValue": msol.get_objective_value(),
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

class Heuristics(js_methods.Heuristics):

    @staticmethod
    def list_heuristic(instance: JmCmax_Instance, rule_number: int = 0, reverse = False) -> RootProblem.SolveResult:
        """contains a list of static dispatching rules to be chosen from

        Args:
            instance (JmCmax_Instance): Instance to be solved
            rule_number (int, optional) : Index of the rule to use. Defaults to 1.

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        default_rule = lambda instance, job_tuple: instance.P[job_tuple[0]][job_tuple[1][0]]
        rules_dict = {
            0: default_rule,
            1: lambda instance, job_tuple: sum(instance.P[job_tuple[0]][oper_idx] for oper_idx in job_tuple[1])
        }
        
        sorting_func = rules_dict.get(rule_number, default_rule)

        return Heuristics.dispatch_heuristic(instance, sorting_func, reverse)

    @staticmethod
    def shifting_bottleneck(instance : JmCmax_Instance):
        """Shifting bottleneck heuristic, Pinedo page 193

        Args:
            instance (JmCmax_Instance): Instance to be solved

        Returns:
            RootProblem.SolveResult: SolveResult of the instance by the method
        """
        startTime = perf_counter()
        solution = JobShop.JobShopSolution(instance)
        solution.create_solution_graph()
        Cmax = solution.graph.critical_path()
        remaining_machines = list(range(instance.m))
        scheduled_machines = []
        precedence_constraints = [] # Tuple of (job_i_id, job_j_id) with job_i preceding job_j

        while len(remaining_machines)>0:
            Cmax = solution.graph.critical_path()
            machines_schedule = []
            taken_solution = None
            objective_value = None
            taken_machine = None
            edges_to_add = None
            for machine in remaining_machines:
                
                vertices = [op[1] for op in solution.graph.get_operations_on_machine(machine)]
                job_id_mapping = {i:vertices[i] for i in range(len(vertices))}
                mapped_constraints =[]
                for precedence in precedence_constraints :
                    if precedence[0] in vertices and precedence[1] in vertices :
                        mapped_constraints.append((list(job_id_mapping.keys())
                            [list(job_id_mapping.values()).index(precedence[0])],list(job_id_mapping.keys())
                            [list(job_id_mapping.values()).index(precedence[1])]))
                Lmax_instance = solution.graph.generate_riPrecLmax(machine,Cmax,mapped_constraints)
                
                BB = JobShop.riPrecLmax.BB(Lmax_instance)
                BB.solve()
                mapped_IDs_solution = [JobShop.Job(job_id_mapping[job.id],job.start_time,job.end_time) for job in BB.best_solution.machine.job_schedule]
                if objective_value is None or objective_value < BB.objective_value:
                    objective_value = BB.objective_value
                    taken_solution = mapped_IDs_solution
                    taken_machine = machine
                    edges_to_add = [((machine,mapped_IDs_solution[ind].id),(machine,mapped_IDs_solution[ind+1].id)) for ind in range(len(BB.best_solution.machine.job_schedule)-1)]


            remaining_machines.remove(taken_machine)
            scheduled_machines.append(taken_machine)
            solution.machines[taken_machine].job_schedule = taken_solution
            solution.machines[taken_machine].objective = taken_solution[len(taken_solution)-1].end_time
            solution.graph.add_disdjunctive_arcs(instance,edges_to_add)
            precedence_constraints = list(solution.graph.generate_precedence_constraints(remaining_machines))
            solution.objective_value = solution.graph.critical_path()

        solution.compute_objective()
        
        return RootProblem.SolveResult(best_solution=solution,status=RootProblem.SolveStatus.FEASIBLE,runtime=perf_counter()-startTime,solutions=[solution])

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]