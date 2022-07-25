import random
import sys
from dataclasses import dataclass, field
from math import exp
from pathlib import Path
from random import randint, uniform
from statistics import mean
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice as np_choice

import pyscheduling_cc.ParallelMachines as ParallelMachines
import pyscheduling_cc.Problem as Problem
from pyscheduling_cc.Problem import Solver


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
        f.write(str(self.n)+" "+str(self.m)+"\n")
        f.write(str(self.m)+"\n")
        for i in range(self.n):
            for j in range(self.m):
                f.write(str(self.P[i][j])+"\t")
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
class RmSijkCmax_Solution(ParallelMachines.ParallelSolution):

    def __init__(self, instance: RmSijkCmax_Instance = None, configuration: list[ParallelMachines.Machine] = None, objective_value: int = 0):
        """Constructor of RmSijkCmax_Solution

        Args:
            instance (RmSijkCmax_Instance, optional): Instance to be solved by the solution. Defaults to None.
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

    def copy(self):
        copy_machines = []
        for m in self.configuration:
            copy_machines.append(m.copy())

        copy_solution = RmSijkCmax_Solution(self.instance)
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
                    expected_start_time = 0
                else:
                    setup_time = self.instance.S[machine.machine_num][prev_job][job]
                    expected_start_time = ci

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
        solution = RmSijkCmax_Solution(instance=instance)
        remaining_jobs_list = [i for i in range(instance.n)]
        while len(remaining_jobs_list) != 0:
            min_factor = None
            for i in remaining_jobs_list:
                for j in range(instance.m):
                    current_machine_schedule = solution.configuration[j]
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
            if (solution.configuration[taken_machine].last_job == -1):
                ci = solution.configuration[taken_machine].completion_time + \
                    instance.P[taken_job][taken_machine]
            else:
                ci = solution.configuration[taken_machine].completion_time + instance.P[taken_job][taken_machine] + \
                    instance.S[taken_machine][solution.configuration[taken_machine].last_job][taken_job]

            solution.configuration[taken_machine].job_schedule.append(ParallelMachines.Job(
                taken_job, solution.configuration[taken_machine].completion_time, ci))
            solution.configuration[taken_machine].completion_time = ci
            solution.configuration[taken_machine].last_job = taken_job

            remaining_jobs_list.remove(taken_job)
            if (ci > solution.objective_value):
                solution.objective_value = ci

        return Problem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

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
        solution = RmSijkCmax_Solution(instance=instance)
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
                current_machine_schedule = solution.configuration[j]
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

            if (solution.configuration[taken_machine].last_job == -1):
                ci = solution.configuration[taken_machine].completion_time + \
                    instance.P[taken_job][taken_machine]
            else:
                ci = solution.configuration[taken_machine].completion_time + instance.P[taken_job][taken_machine] + \
                    instance.S[taken_machine][solution.configuration[taken_machine].last_job][taken_job]
            solution.configuration[taken_machine].job_schedule.append(ParallelMachines.Job(
                taken_job, solution.configuration[taken_machine].completion_time, ci))
            solution.configuration[taken_machine].completion_time = ci
            solution.configuration[taken_machine].last_job = taken_job
            if (ci > solution.objective_value):
                solution.objective_value = ci
        return Problem.SolveResult(best_solution=solution, runtime=perf_counter()-start_time, solutions=[solution])

    @classmethod
    def all_methods(cls):
        """returns all the methods of the given Heuristics class

        Returns:
            list[object]: list of functions
        """
        return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and not func == "all_methods"]


class Metaheuristics():

    @staticmethod
    def meta_raps(instance: RmSijkCmax_Instance, p: float, r: int, nb_exec: int):
        """Returns the solution using the meta-raps algorithm

        Args:
            instance (RmSijkCmax_Instance): The instance to be solved by the metaheuristic
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
            solution = RmSijkCmax_Solution(instance)
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
    def grasp(instance: RmSijkCmax_Instance, x, nb_exec: int):
        """Returns the solution using the grasp algorithm

        Args:
            instance (RmSijkCmax_Instance): Instance to be solved by the metaheuristic
            x (_type_): percentage of moves to consider to select the best move
            nb_exec (int): Number of execution of the metaheuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = Problem.SolveResult()
        solveResult.all_solutions = []
        best_solution = None
        for _ in range(nb_exec):
            solution = RmSijkCmax_Solution(instance)
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

    @staticmethod
    def antColony(instance: RmSijkCmax_Instance, **data):
        """Returns the solution using the ant colony algorithm

        Args:
            instance (RmSijkCmax_Instance): Instance to be solved

        Returns:
            Problem.SolveResult: the solver result of the execution of the metaheuristic
        """
        startTime = perf_counter()
        solveResult = Problem.SolveResult()
        AC = AntColony(instance=instance, **data)
        solveResult.best_solution, solveResult.all_solutions = AC.solve()
        solveResult.solve_status = Problem.SolveStatus.FEASIBLE
        solveResult.runtime = perf_counter() - startTime
        return solveResult

    @staticmethod
    def lahc(instance: RmSijkCmax_Instance, **kwargs):
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
    def SA(instance: RmSijkCmax_Instance, **kwargs):
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

    def spread_pheronome_global(self, all_solutions: list[RmSijkCmax_Solution]):
        """Update pheromone levels globally after finding new solutions

        Args:
            all_solutions (list[RmSijkCmax_Solution]): list of generated solutions
        """
        sorted_solutions = sorted(all_solutions, key=lambda x: x[1])

        for solution, cmax_i in sorted_solutions[:self.n_best]:
            for k in range(solution.instance.m):
                machine_k = solution.configuration[k]
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
            RmSijkCmax_Solution: complete solution of one ant
        """
        pheromone = self.aco_graph[1]
        solution_path = RmSijkCmax_Solution(self.instance)

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

            current_machine = solution_path.configuration[m]
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
