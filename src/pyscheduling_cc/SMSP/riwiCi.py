import random
import sys
from dataclasses import dataclass, field
from random import randint, uniform
from pathlib import Path
from time import perf_counter

from matplotlib import pyplot as plt

import pyscheduling_cc.Problem as Problem
from pyscheduling_cc.Problem import Solver
import pyscheduling_cc.SMSP.SingleMachine as SingleMachine
from pyscheduling_cc.SMSP.SingleMachine import ExactSolvers


@dataclass
class riwiCi_Instance(SingleMachine.SingleInstance):
    W : list[int] = field(default_factory=list) # Jobs weights
    P: list[int] = field(default_factory=list)  # Processing time
    R: list[int] = field(default_factory=list)

    @classmethod
    def read_txt(cls, path: Path):
        """Read an instance from a txt file according to the problem's format

        Args:
            path (Path): path to the txt file of type Path from the pathlib module

        Raises:
            FileNotFoundError: when the file does not exist

        Returns:
            wiCi_Instance:

        """
        f = open(path, "r")
        content = f.read().split('\n')
        ligne0 = content[0].split(' ')
        n = int(ligne0[0])  # number of jobs
        i = 1
        instance = cls("test", n)
        instance.P, i = instance.read_P(content, i)
        instance.W, i = instance.read_W(content, i)
        instance.R, i = instance.read_R(content, i)
        f.close()
        return instance

    @classmethod
    def generate_random(cls, jobs_number: int,  protocol: SingleMachine.GenerationProtocol = SingleMachine.GenerationProtocol.BASE, law: SingleMachine.GenerationLaw = SingleMachine.GenerationLaw.UNIFORM, Wmin : int = 1, Wmax : int = 1 ,Pmin: int = 1, Pmax: int = -1, Alpha: float = 0.0, InstanceName: str = ""):
        """Random generation of RmSijkCmax problem instance

        Args:
            jobs_number (int): number of jobs of the instance
            protocol (SingleMachine.GenerationProtocol, optional): given protocol of generation of random instances. Defaults to SingleMachine.GenerationProtocol.VALLADA.
            law (SingleMachine.GenerationLaw, optional): probablistic law of generation. Defaults to SingleMachine.GenerationLaw.UNIFORM.
            Pmin (int, optional): Minimal processing time. Defaults to -1.
            Pmax (int, optional): Maximal processing time. Defaults to -1.
            Alpha (float,optional): Release time factor. Defaults to 0.0.
            InstanceName (str, optional): name to give to the instance. Defaults to "".

        Returns:
            wiCi_Instance: the randomly generated instance
        """
        if(Pmax == -1):
            Pmax = randint(Pmin, 100)
        if(Alpha == 0.0):
            Alpha = round(uniform(1.0, 3.0), 1)
        instance = cls(InstanceName, jobs_number)
        instance.P = instance.generate_P(protocol, law, Pmin, Pmax)
        instance.W = instance.generate_W(protocol,law, Wmin, Wmax)
        instance.R = instance.generate_R(
            protocol, law, instance.P, Pmin, Pmax, Alpha)
        return instance

    def to_txt(self, path: Path) -> None:
        """Export an instance to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(str(self.n))
        f.write("\nProcessing time\n")
        for i in range(self.n):
            f.write(str(self.P[i])+"\t")
        f.write("\nWeights\n")
        for i in range(self.n):
            f.write(str(self.W[i])+"\t")
        f.write("\nRelease time\n")
        for i in range(self.n):
            f.write(str(self.R[i])+"\t")
        f.close()

    def create_solution(self):
        return riwiCi_Solution(self)


@dataclass
class riwiCi_Solution(SingleMachine.SingleSolution):

    def __init__(self, instance: riwiCi_Instance = None, machine : SingleMachine.Machine = None, objective_value: int = 0):
        """Constructor of riwiCi_Solution

        Args:
            instance (wiCi_Instance, optional): Instance to be solved by the solution. Defaults to None.
            configuration (SingleMachine.Machine, optional): list of machines of the instance. Defaults to None.
            objective_value (int, optional): initial objective value of the solution. Defaults to 0.
        """
        self.instance = instance
        if machine is None:
            self.machine = SingleMachine.Machine(0, -1, [])
        else:
            self.machine = machine
        self.objective_value = objective_value

    def __str__(self):
        return "Cmax : " + str(self.objective_value) + "\n" + "Job_schedule (job_id , start_time , completion_time) | Completion_time\n" + self.machine.__str__()

    def copy(self):
        copy_solution = riwiCi_Solution(self.instance)
        copy_solution.machine = self.machine.copy()
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
        line_content = content[2].split('|')
        machine = SingleMachine.Machine(int(line_content[1]), job_schedule=[SingleMachine.Job(
                int(j[0]), int(j[1]), int(j[2])) for j in [job.strip()[1:len(job.strip())-1].split(',') for job in line_content[0].split(':')]])
        solution = cls(objective_value=objective_value_,
                       machine=machine)
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
                ticks.append(15)

                gnt.set_yticks(ticks)
                # Labelling tickes of y-axis
                gnt.set_yticklabels(ticks_labels)

                # Setting graph attribute
                gnt.grid(True)

                schedule = self.machine.job_schedule
                prevEndTime = 0
                for element in schedule:
                    job_index, startTime, endTime = element
                    if prevEndTime < startTime:
                        # Idle Time
                        gnt.broken_barh(
                            [(prevEndTime, startTime - prevEndTime)], (10, 9), facecolors=('tab:gray'))
                    
                    gnt.broken_barh([(startTime, self.instance.P[job_index])], (
                        10, 9), facecolors=('tab:blue'))
                    
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
        ci, expected_start_time = 0, 0
        for i, element in enumerate(self.machine.job_schedule):
            job, startTime, endTime = element
            # Test End Time + start Time
            expected_start_time = ci
            ci +=  self.instance.P[job]

            if startTime != expected_start_time or endTime != ci:
                print(f'## Error:  found {element} expected {job,expected_start_time, ci}')
                is_valid = False
            set_jobs.add(job)

        is_valid &= len(set_jobs) == self.instance.n
        return is_valid


class Heuristics():
    @staticmethod
    def WSECi(instance : riwiCi_Instance):
        startTime = perf_counter()
        solution = riwiCi_Solution(instance)
        solution.machine.wiCi_index = []
        ci = 0
        wiCi = 0
        remaining_jobs_list = list(range(instance.n))
        rule = lambda job_id : float(instance.W[job_id])/float(max(instance.R[job_id] - ci,0) + instance.P[job_id])
        while(len(remaining_jobs_list)>0):
            remaining_jobs_list.sort(reverse=True,key=rule)
            taken_job = remaining_jobs_list[0]
            start_time = max(instance.R[taken_job],ci)
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,start_time,start_time+instance.P[taken_job]))
            ci = start_time+instance.P[taken_job]
            wiCi += instance.W[taken_job]*ci
            solution.machine.wiCi_index.append(wiCi)
            remaining_jobs_list.pop(0)
        solution.machine.objective=solution.machine.wiCi_index[instance.n-1]
        solution.fix_objective()
        return Problem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])

    @staticmethod
    def WSAPT(instance : riwiCi_Instance):
        startTime = perf_counter()
        solution = riwiCi_Solution(instance)
        solution.machine.wiCi_index = []
        ci = min(instance.R)
        wiCi = 0
        remaining_jobs_list = list(range(instance.n))
        
        rule = lambda job_id : float(instance.W[job_id])/float(instance.P[job_id])
        while(len(remaining_jobs_list)>0):
            filtered_remaining_jobs_list = list(filter(lambda job_id : instance.R[job_id]<=ci,remaining_jobs_list))
            filtered_remaining_jobs_list.sort(reverse=True,key=rule)
            if(len(filtered_remaining_jobs_list)==0):
                ci = min([instance.R[job_id] for job_id in remaining_jobs_list])
                filtered_remaining_jobs_list = list(filter(lambda job_id : instance.R[job_id]<=ci,remaining_jobs_list))
                filtered_remaining_jobs_list.sort(reverse=True,key=rule)

            taken_job = filtered_remaining_jobs_list[0]
            start_time = max(instance.R[taken_job],ci)
            ci = start_time+instance.P[taken_job]
            solution.machine.job_schedule.append(SingleMachine.Job(taken_job,start_time,ci))
            wiCi += instance.W[taken_job]*ci
            solution.machine.wiCi_index.append(wiCi)
            remaining_jobs_list.remove(taken_job)

        solution.machine.objective=solution.machine.wiCi_index[instance.n-1]
        solution.fix_objective()
        return Problem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])

    @staticmethod
    def list_heuristic(instance : riwiCi_Instance, rule : int = 1):

        startTime = perf_counter()
        solution = riwiCi_Solution(instance)
        solution.machine.wiCi_index = []
        if rule==1: # Increasing order of the release time
            sorting_func = lambda job_id : instance.R[job_id]
            reverse = False
        elif rule==2: # WSPT
            sorting_func = lambda job_id : float(instance.W[job_id])/float(instance.P[job_id])
            reverse = True
        elif rule ==3: #WSPT including release time in the processing time
            sorting_func = lambda job_id : float(instance.W[job_id])/float(instance.R[job_id]+instance.P[job_id])
            reverse = True

        remaining_jobs_list = list(range(instance.n))
        remaining_jobs_list.sort(reverse=reverse,key=sorting_func)
        
        ci = 0
        wiCi = 0
        for job in remaining_jobs_list:
            start_time = max(instance.R[job],ci)
            ci = start_time + instance.P[job]
            wiCi += instance.W[job]*ci
            solution.machine.job_schedule.append(SingleMachine.Job(job,start_time,ci))
            solution.machine.wiCi_index.append(wiCi)
        solution.machine.objective = solution.machine.wiCi_index[instance.n - 1]
        solution.fix_objective()
        return Problem.SolveResult(best_solution=solution,runtime=perf_counter()-startTime,solutions=[solution])

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]


class Metaheuristics():

    @staticmethod
    def lahc(instance : riwiCi_Instance, **kwargs):
        """ Returns the solution using the LAHC algorithm
        Args:
            instance (riwiCi_Instance): Instance object to solve
            Lfa (int, optional): Size of the candidates list. Defaults to 25.
            Nb_iter (int, optional): Number of iterations of LAHC. Defaults to 300.
            Non_improv (int, optional): LAHC stops when the number of iterations without
                improvement is achieved. Defaults to 50.
            LS (bool, optional): Flag to apply local search at each iteration or not.
                Defaults to True.
            time_limit_factor: Fixes a time limit as follows: n*m*time_limit_factor if specified, 
                else Nb_iter is taken Defaults to None
            init_sol_method: The method used to get the initial solution. 
                Defaults to "WSECi"
            seed (int, optional): Seed for the random operators to make the algo deterministic
        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """

        # Extracting parameters
        time_limit_factor = kwargs.get("time_limit_factor", None)
        init_sol_method = kwargs.get("init_sol_method", Heuristics.WSECi)
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

        local_search = SingleMachine.SM_LocalSearch()

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

            solution_i = SingleMachine.NeighbourhoodGeneration.lahc_neighbour(
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
    def iterative_LS(instance : riwiCi_Instance, ** kwargs):
        time_limit_factor = kwargs.get("time_limit_factor", None)
        init_sol_method = kwargs.get("init_sol_method", Heuristics.WSECi)
        Nb_iter = kwargs.get("Nb_iter", 500000)
        Non_improv = kwargs.get("Non_improv", 50000)

        first_time = perf_counter()
        if time_limit_factor:
            time_limit = instance.m * instance.n * time_limit_factor

        # Generate init solutoin using the initial solution method
        solution_init = init_sol_method(instance).best_solution

        if not solution_init:
            return Problem.SolveResult()

        local_search = SingleMachine.SM_LocalSearch()

        all_solutions = []
        solution_best = solution_init.copy()  # Save the current best solution
        solution_i = solution_init.copy()
        all_solutions.append(solution_best)

        N = 0
        i = 0
        time_to_best = perf_counter() - first_time
        while i < Nb_iter and N < Non_improv:
            # check time limit if exists
            if time_limit_factor and (perf_counter() - first_time) >= time_limit:
                break

            solution_i = local_search.improve(solution_i)

            if solution_i.objective_value < solution_best.objective_value:
                all_solutions.append(solution_i)
                solution_best = solution_i.copy()
                time_to_best = (perf_counter() - first_time)
                N = 0
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

    
    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]
