import random
import sys
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np

import pyscheduling.Problem as Problem
from pyscheduling.BaseConstraints import BaseConstraints as Constraints
from pyscheduling.Problem import Job, Objective
from pyscheduling.SMSP.SingleMachine import Machine as SMachine


class GenerationProtocol(Enum):
    BASE = 1
    VALLADA = 2

@dataclass
class ParallelInstance(Problem.BaseInstance):

    n: int  # n : Number of jobs
    m: int  # m : Number of machines

    def __init__(self, n: int, m: int, name: str = "Unknown", **kwargs):
        super().__init__(n, m=m, name = name, **kwargs)

@dataclass
class Machine(SMachine):
    
    machine_num:int = 0 
    
    def __init__(self, machine_num:int, **kwargs):
        super().__init__(**kwargs)
        self.machine_num = machine_num
    
    def __repr__(self):
        return f"Machine {self.machine_num} : " + " -> ".join(map(str, [(job.id, job.start_time, job.end_time)\
                                    for job in self.job_schedule])) +\
                                    " | Objective value: " + str(self.objective_value)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        wiCi_copy = self.wiCi_cache if self.wiCi_cache is None else list(self.wiCi_cache)
        wiTi_copy = self.wiTi_cache if self.wiTi_cache is None else list(self.wiTi_cache)
        wiFi_copy = self.wiFi_cache if self.wiFi_cache is None else list(self.wiFi_cache)
        return Machine(self.machine_num, objective_value = self.objective_value, last_job = self.last_job, job_schedule = list(self.job_schedule),
                        wiCi_cache=wiCi_copy, wiTi_cache=wiTi_copy, wiFi_cache=wiFi_copy)
    
    def compute_current_ci(self, instance: ParallelInstance, prev_ci: int, job_prev_i: int, job_i: int):
        """Computes the current ci when job_i comes after job_prev_i.
        This takes into account if we have setup times and release dates.

        Args:
            instance (SingelInstance): the instance to be solved.
            prev_ci (int): the previous value of ci
            job_prev_i (int): id of the job that precedes the inserted job 
            job_i (int): id of the job to be inserted at the end

        Returns:
            tuple: (ci, start_time), the new completion time and start_time of the inserted job.
        """
        startTime = max(prev_ci, instance.R[job_i]) if hasattr(instance, 'R') else prev_ci
        setupTime = instance.S[self.machine_num][job_prev_i][job_i] if hasattr(instance, 'S') else 0
        proc_time = instance.P[job_i][self.machine_num]

        ci = startTime + setupTime + proc_time
        
        return ci, startTime

@dataclass
class ParallelSolution(Problem.BaseSolution):

    machines: List[Machine]
    # Class variables
    max_objectives = {Objective.Cmax, Objective.Lmax}
    sum_objectives = {Objective.wiCi, Objective.wiTi, Objective.wiFi}

    def __init__(self, instance: ParallelInstance = None, machines: List[Machine] = None, objective_value: int = 0):
        """Constructor of RmSijkCmax_Solution

        Args:
            instance (ParallelInstance, optional): Instance to be solved by the solution. Defaults to None.
            configuration (list[ParallelMachines.Machine], optional): list of machines of the instance. Defaults to None.
            objective_value (int, optional): initial objective value of the solution. Defaults to 0.
        """
        self.instance = instance
        if machines is None:
            self.machines = []
            for i in range(instance.m):
                machine = Machine(i)
                self.machines.append(machine)
        else:
            self.machines = machines
        self.objective_value = 0

    def __repr__(self):
        return "Objective : " + str(self.objective_value) + "\n\n" + "Machine_ID | Job_schedule (job_id , start_time , completion_time) | Objective value\n\n" + "\n\n####\n\n".join(map(str, self.machines))

    def __str__(self):
        return self.__repr__()

    def copy(self):
        copy_machines = []
        for m in self.machines:
            copy_machines.append(m.copy())

        copy_solution = ParallelSolution(self.instance)
        for i in range(self.instance.m):
            copy_solution.machines[i] = copy_machines[i]
        copy_solution.objective_value = self.objective_value
        return copy_solution

    def __lt__(self, other):
        if self.instance.get_objective().value > 0 :
            return self.objective_value < other.objective_value
        else : return other.objective_value < self.objective_value

    def tmp_objective(self, tmp_obj=None):
        """returns the temporary objective_value of a solution according to the the machines objectives from the dict temp_obj if present, 
        if not it takes the objective_value of the machine, this doesn't modify the "cmax" of the machine.
        
        Args:
            temp_obj (dict, optional): temporary objectives for each machine, machine_num: tmp_obj. Defaults to None.

        Returns:
            int: tmp_obj
        """
        tmp_obj = tmp_obj if tmp_obj is not None else dict()
        objectives_list = [tmp_obj.get(machine.machine_num, machine.objective_value) for machine in self.machines ]
        objective = self.instance.get_objective()
        if objective in self.max_objectives:
            return max(objectives_list)
        elif objective in self.sum_objectives:
            return sum(objectives_list)

    def fix_objective(self):
        """Sets the objective_value of the solution to the correct value
            according to the objective_values of the machines (without computing them)
        """
        objective = self.instance.get_objective()
        if objective in self.max_objectives:
            self.objective_value = max(machine.objective_value for machine in self.machines)
        elif objective in self.sum_objectives:
            self.objective_value = sum(machine.objective_value for machine in self.machines)

        return self.objective_value
    
    def compute_objective(self,instance):
        """Computes the current solution's objective.
            By calling the compute objective on each machine.
        """ 
        for machine in self.machines:
            machine.compute_objective(instance)          

        return self.fix_objective()
    
    def fix_solution(self):
        for machine in self.machines:
            if len(machine.job_schedule) > 0:
                machine.last_job = machine.job_schedule[len(machine.job_schedule)-1][0]
        
        return 
                
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
            configuration_.append(Machine(int(line_content[0]), int(line_content[2]), job_schedule=[Job(
                int(j[0]), int(j[1]), int(j[2])) for j in [job.strip()[1:len(job.strip())-1].split(',') for job in line_content[1].split(':')]]))
        solution = cls(objective_value=objective_value_,
                       machines=configuration_)
        return solution

    def to_txt(self, path: Path) -> None:
        """Export the solution to a txt file

        Args:
            path (Path): path to the resulting txt file
        """
        f = open(path, "w")
        f.write(self.__str__())
        f.close()

    def plot(self, path: Path = None):
        """Plot the solution into a gantt diagramm

        Args:
            types (str, optional): The type of tasks to plot, a string where each letter corresponds to one type: R for release, \n
                S for setup and P for Processing. Defaults to "RSP".
            path (Path, optional): The path to export the diagram, if not specified it is not exported but shown inline. Defaults to None.
        """
        tasks_df = []
        cmax_value = 0
        for j in range(len(self.machines)):
            schedule = self.machines[j].job_schedule
            prev = schedule[0].id # First job in the schedule
            prevEndTime = 0
            for element in schedule:
                job_index, startTime, endTime = element
                
                # Idle time
                if prevEndTime < startTime:
                    idle_time_task = dict(Task=f"M{j}", Description=f"J{job_index} idle", Start=prevEndTime,
                                        Finish = startTime, Type = 'Idle')
                    tasks_df.append(idle_time_task)
                
                # Setup time
                end_setup = startTime + self.instance.S[j][prev][job_index] if hasattr(self.instance, "S") else startTime
                if startTime != end_setup:
                    setup_task = dict(Task=f"M{j}", Description=f"J{job_index} setup",
                                    Start = startTime, Finish = end_setup, Type = 'Setup')
                    tasks_df.append(setup_task)
                
                # Proc time
                end_proc = end_setup + self.instance.P[job_index][j]
                proc_task = dict(Task=f"M{j}", Description=f"J{job_index} proc", 
                                Start = end_setup, Finish = end_proc, Type = 'Processing')
                tasks_df.append(proc_task)
                
                prev = job_index
                prevEndTime = endTime
                cmax_value = max(cmax_value, endTime)

        self._plot_tasks(tasks_df, path)
        

    def is_valid(self):
        """
        # Check if solution respects the constraints
        """
        set_jobs = set()
        is_valid = True
        obj_list = []
        obj_sol = 0
        objective = self.instance.get_objective()
        for machine in self.machines:
            prev_job = None
            expected_start_time, setup_time, ci  = 0, 0, 0
            obj_machine = 0
            for i, job_element in enumerate(machine.job_schedule):
                job_id, startTime, endTime = job_element
                # Test End Time + start Time
                
                # Check release constraint
                if hasattr(self.instance,'R'):
                    expected_start_time = max(self.instance.R[job_id],ci)
                else:
                    expected_start_time = ci
                
                # Check setup constraint
                if hasattr(self.instance,'S'):
                    if prev_job is None:
                        setup_time = self.instance.S[machine.machine_num][job_id][job_id]
                    else:
                        setup_time = self.instance.S[machine.machine_num][prev_job][job_id]
                else: 
                    setup_time = 0

                proc_time = self.instance.P[job_id][machine.machine_num]
                ci = expected_start_time + setup_time + proc_time 

                if startTime != expected_start_time or endTime != ci:
                    print(f'## Error: in machine {machine.machine_num}' +
                          f' found {job_element} expected {job_id,expected_start_time, ci}')
                    is_valid = False
                
                if objective == Objective.Cmax:
                    obj_machine = ci
                elif objective == Objective.wiCi:
                    obj_machine += self.instance.W[job_id] * ci
                elif objective == Objective.wiTi:
                    obj_machine += self.instance.W[job_id] * max(ci-self.instance.D[job_id], 0)
                elif objective == Objective.wiFi:
                    obj_machine += self.instance.W[job_id] * (ci-self.instance.R[job_id])
                elif objective == Objective.Lmax:
                    obj_machine = max(obj_machine, ci - self.instance.D[job_id])

                set_jobs.add(job_id)
                prev_job = job_id
            
            if obj_machine != machine.objective_value:
                print(f'## Error: in machine {machine.machine_num}' +
                    f' found objective_value = {machine.objective_value} expected {obj_machine}')
                is_valid = False

            obj_list.append(obj_machine)

        if objective in self.max_objectives:
            obj_sol = max(obj_list)
        elif objective in self.sum_objectives:
            obj_sol = sum(obj_list)

        if obj_sol != self.objective_value:
            print(f'## Error: in solution' +
                    f' found objective_value = {self.objective_value} expected {obj_sol}')
            is_valid = False
        
        if len(set_jobs) != self.instance.n:
            print(f'## Error: in number of jobs' +
                    f' found {len(set_jobs)} job(s) expected {self.instance.n}')
            is_valid = False

        return is_valid


class PM_LocalSearch(Problem.LocalSearch):
    @staticmethod
    def _external_insertion(solution: ParallelSolution):
        """Delete a job from the machine whose completion_time is maximal and insert it on another one

        Args:
            solution (ParallelSolution): The initial solution to be improved

        Returns:
            ParallelSolution: Improved solution
        """
        bottleneck_machines_list, other_machines_list = \
            PM_LocalSearch.get_bottleneck_machines(solution)
        
        for nb_machine in bottleneck_machines_list:
            bottleneck_machine = solution.machines[nb_machine]
            bottleneck_machine_schedule = bottleneck_machine.job_schedule
            old_obj_i = bottleneck_machine.objective_value
            if len(bottleneck_machine_schedule) < 2: # If it only has one job, don't remove it
                continue

            move = None
            other_machines_copy = list(other_machines_list)
            while move is None and len(other_machines_copy) != 0:
                random_index = random.randrange(len(other_machines_copy))
                other_machine_index = other_machines_copy.pop(random_index)
                other_machine = solution.machines[other_machine_index]
                other_nb_machine = other_machine.machine_num
                other_machine_schedule = other_machine.job_schedule

                old_obj_l = other_machine.objective_value
                old_obj = solution.objective_value
                best_obj = old_obj
                best_diff = None
                for j in range(len(bottleneck_machine_schedule)):
                    job_j, _, _ = bottleneck_machine_schedule[j]
                    obj_i = bottleneck_machine.simulate_remove_insert(j, -1, -1, solution.instance)
                    for k in range(len(other_machine_schedule)):
                        obj_l = other_machine.simulate_remove_insert(-1, job_j,k,solution.instance)
                        new_obj = solution.tmp_objective(tmp_obj= {nb_machine: obj_i, other_nb_machine: obj_l})
                        potential_move = (other_machine_index, j, k, obj_i, obj_l)
                        if new_obj < old_obj:
                            move = potential_move if move is None or new_obj < best_obj else move 
                            best_obj = new_obj if new_obj < best_obj else best_obj
                        elif new_obj == best_obj and (obj_i < old_obj_i or obj_l < old_obj_l):
                            cond = (best_diff is None or old_obj_i - obj_i + old_obj_l - obj_l < best_diff) and best_obj == old_obj
                            move = potential_move if move is None or cond else move
                            best_diff = old_obj_i - obj_i + old_obj_l - obj_l if best_diff is None or cond else best_diff

            if move:  # Apply the best move
                other_machine_index, j, k, obj_i, obj_l = move
                other_machine = solution.machines[other_machine_index]
                taken_job = bottleneck_machine_schedule.pop(j)
                other_machine.job_schedule.insert(k, taken_job)

                bottleneck_machine.compute_objective(solution.instance, startIndex=0)
                other_machine.compute_objective(solution.instance, startIndex=0)
                solution.fix_objective()
        
        return solution
    
    @staticmethod
    def _external_swap(solution: ParallelSolution):
        """Swap between 2 jobs on different machines, where one of the machines has the maximal completion_time among all

        Args:
            solution (ParallelSolution): The initial solution to be improved

        Returns:
            ParallelSolution: Improved solution
        """
        bottleneck_machines_list, other_machines_list = \
            PM_LocalSearch.get_bottleneck_machines(solution)

        for nb_machine in bottleneck_machines_list:
            bottleneck_machine = solution.machines[nb_machine]
            bottleneck_machine_schedule = bottleneck_machine.job_schedule
            old_obj_i = bottleneck_machine.objective_value

            move = None
            other_machines_copy = list(other_machines_list)
            while move is None and len(other_machines_copy) != 0:
                random_index = random.randrange(len(other_machines_copy))
                other_machine_index = other_machines_copy.pop(random_index)
                other_machine = solution.machines[other_machine_index]
                other_nb_machine = other_machine.machine_num
                other_machine_schedule = other_machine.job_schedule

                old_obj_l = other_machine.objective_value
                old_obj = solution.objective_value
                best_obj = old_obj
                best_diff = None
                for j in range(len(bottleneck_machine_schedule)):
                    for k in range(len(other_machine_schedule)):
                        job_j, _, _ = bottleneck_machine_schedule[j]
                        job_k, _, _ = other_machine_schedule[k]

                        obj_i = bottleneck_machine.simulate_remove_insert(j, job_k, j, solution.instance)
                        obj_l = other_machine.simulate_remove_insert(k, job_j, k, solution.instance)
                        
                        new_obj = solution.tmp_objective(tmp_obj = {nb_machine: obj_i, other_nb_machine: obj_l})
                        potential_move = (other_machine_index, j, k, obj_i, obj_l)
                        if new_obj < old_obj:
                            move = potential_move if move is None or new_obj < best_obj else move 
                            best_obj = new_obj if new_obj < best_obj else best_obj
                        elif new_obj == best_obj and (obj_i < old_obj_i or obj_l < old_obj_l):
                            cond = (best_diff is None or old_obj_i - obj_i + old_obj_l - obj_l < best_diff) and best_obj == old_obj
                            move = potential_move if move is None or cond else move
                            best_diff = old_obj_i - obj_i + old_obj_l - obj_l if best_diff is None or cond else best_diff

            if move:  # Apply the best move
                other_machine_index, j, k, obj_i, obj_l = move
                other_machine = solution.machines[other_machine_index]
                other_machine_schedule = other_machine.job_schedule
                bottleneck_machine_schedule[j],  other_machine_schedule[k] = other_machine_schedule[k], bottleneck_machine_schedule[j]
                
                bottleneck_machine.compute_objective(solution.instance,startIndex=0)
                other_machine.compute_objective(solution.instance,0)
                solution.fix_objective()
                
        return solution
    
    @staticmethod
    def _internal_swap(solution: ParallelSolution):
        """Swap between 2 jobs on the same machine whose completion_time is maximal if it gives a better solution

        Args:
            solution (ParallelSolution): The initial solution to be improved

        Returns:
            ParallelSolution: Improved solution
        """
        objective = solution.instance.get_objective()
        taken_machines_list = []
        for m, machine in enumerate(solution.machines):
            if machine.objective_value == solution.objective_value or objective not in solution.max_objectives:
                taken_machines_list.append(m)

        for nb_machine in taken_machines_list:
            bottleneck_machine = solution.machines[nb_machine]
            bottleneck_machine_schedule = bottleneck_machine.job_schedule
            move = None
            for i in range(0, len(bottleneck_machine_schedule)):
                for j in range(i+1, len(bottleneck_machine_schedule)):
                    new_obj = bottleneck_machine.simulate_swap(i, j, solution.instance)
                    if new_obj < bottleneck_machine.objective_value:
                        move = (i, j, new_obj) if move is None or new_obj < move[2] else move
 
            if move:
                i, j, new_obj = move
                bottleneck_machine_schedule[i], bottleneck_machine_schedule[j] = bottleneck_machine_schedule[j], bottleneck_machine_schedule[i]
                bottleneck_machine.compute_objective(solution.instance, startIndex=0)
                solution.fix_objective()

        return solution
    
    @staticmethod
    def _inter_machine_insertion(solution: ParallelSolution):
        """For every job, verify if rescheduling it on the same machine at a different position or on a whole different machines gives a better solution

        Args:
            solution (ParallelSolution): The initial solution to be improved

        Returns:
            ParallelSolution: Improved solution
        """
        for i in range(solution.instance.m):  # For every machine in the system
            for l in range(solution.instance.m):  # For all the other machines
                if (l == i) or len(solution.machines[i].job_schedule) < 2:
                    continue
                
                move = None
                # Machine i
                machine_i = solution.machines[i]
                machine_i_schedule = machine_i.job_schedule
                old_obj_i = machine_i.objective_value
                # Machine L
                machine_l = solution.machines[l]
                machine_l_schedule = machine_l.job_schedule
                old_obj_l = machine_l.objective_value
                # for every job in the machine
                for k in range(len(machine_i_schedule)):
                    job_k = machine_i_schedule[k]
                    obj_i = machine_i.simulate_remove_insert(k, -1, -1, solution.instance)
                    for j in range(len(machine_l_schedule)):
                        obj_l = machine_l.simulate_remove_insert(-1, job_k.id,j,solution.instance)

                        if old_obj_i - obj_i >= obj_l - old_obj_l and obj_l <= solution.objective_value:
                            move = (l, k, j, obj_i, obj_l) if move is None or obj_i-old_obj_i+obj_l-old_obj_l >= obj_i-move[3]+obj_l-move[4] else move

                if move:
                    l, k, j, obj_i, obj_l = move
                    # Remove job k from machine i
                    job_k = machine_i_schedule.pop(k)
                    # Insert job k in machine l in pos j
                    machine_l_schedule.insert(j, job_k)
                    machine_i.compute_objective(solution.instance, startIndex=0)
                    machine_l.compute_objective(solution.instance, startIndex=0)
                    solution.fix_objective()
        
        return solution
    
    @staticmethod
    def _balance(solution: ParallelSolution):
        """Reschedule jobs between machines in order to balance their completion_time thus giving a better solution

        Args:
            solution (ParallelSolution): The initial solution to be improved

        Returns:
            ParallelSolution: Improved solution
        """
        change = True
        while change:
            change = False
            bottleneck_machines_list, other_machines_list = \
                PM_LocalSearch.get_bottleneck_machines(solution)
            
            for nb_machine in bottleneck_machines_list:
                bottleneck_machine = solution.machines[nb_machine]
                bottleneck_machine_schedule = bottleneck_machine.job_schedule
                old_obj_i = bottleneck_machine.objective_value
                if len(bottleneck_machine_schedule) < 2:
                    continue

                other_machines_list.sort(key=lambda m: solution.machines[m].objective_value)
                move = None
                l = 0
                while move is None and l < len(other_machines_list): 
                    other_machine_index = other_machines_list[l]
                    other_machine = solution.machines[other_machine_index]
                    other_nb_machine = other_machine.machine_num
                    other_machine_schedule = other_machine.job_schedule

                    old_obj_l = other_machine.objective_value
                    old_obj = solution.objective_value
                    best_obj = old_obj
                    best_diff = None

                    j = len(bottleneck_machine_schedule) - 1
                    job_j, _, _ = bottleneck_machine_schedule[j]
                    obj_i = bottleneck_machine.simulate_remove_insert(j, -1, -1, solution.instance)
                    for k in range(len(other_machine_schedule)):
                        obj_l = other_machine.simulate_remove_insert(-1, job_j, k, solution.instance)
                        new_obj = solution.tmp_objective(tmp_obj= {nb_machine: obj_i, other_nb_machine: obj_l})
                        potential_move = (other_machine_index, j, k, obj_i, obj_l)
                        if new_obj < old_obj:
                            move = potential_move if move is None or new_obj < best_obj else move 
                            best_obj = new_obj if new_obj < best_obj else best_obj
                        elif new_obj == best_obj and (obj_i < old_obj_i or obj_l < old_obj_l):
                            cond = (best_diff is None or old_obj_i - obj_i + old_obj_l - obj_l < best_diff) and best_obj == old_obj
                            move = potential_move if not move or cond else move
                            best_diff = old_obj_i - obj_i + old_obj_l - obj_l if best_diff is None or cond else best_diff
                    
                    l += 1
                if move:  # Apply the best move
                    change = True
                    other_machine_index, j, k, obj_i, obj_l = move
                    other_machine = solution.machines[other_machine_index]
                    job_j = bottleneck_machine_schedule.pop(j)
                    other_machine.job_schedule.insert(k, job_j)
                    bottleneck_machine.compute_objective(solution.instance, startIndex=0)  
                    other_machine.compute_objective(solution.instance, startIndex=0)
                    solution.fix_objective()
                    
        return solution
    
    @staticmethod
    def best_insertion_machine(solution : ParallelSolution, machine_id : int, job_id : int):
        """Find the best position to insert a job job_id in the machine machine_id

        Args:
            solution (ParallelSolution): Solution to be improved
            machine_id (int): ID of the machine 
            job_id (int): ID of the job

        Returns:
            ParallelSolution: New solution
        """
        machine = solution.machines[machine_id]
        machine_schedule = machine.job_schedule
        best_obj_l = None
        taken_move = 0
        for j in range(len(machine_schedule) + 1):  # for every position in other machine
            obj_l = machine.simulate_remove_insert(-1, job_id, j, solution.instance)

            if best_obj_l is None or obj_l < best_obj_l:
                best_obj_l = obj_l
                taken_move = j

        machine_schedule.insert(taken_move, Job(job_id, 0, 0))
        machine.compute_objective(solution.instance, startIndex=0)

        return solution
     
    @staticmethod
    def get_bottleneck_machines(solution: ParallelSolution):
        """Gets the list of machines that are bottlneck and a list of the remaining machines.
            For the case where the bottlneck is not defined (no max aggregation): 
            half of the machines with the largest objective values is returned as bottlneck.

        Args:
            solution (ParallelSolution): problem solution
        """ 
        objective = solution.instance.get_objective()
        bottleneck_machines_list = []
        other_machines_list = []
        for m, machine in enumerate(solution.machines):
            if machine.objective_value == solution.objective_value or objective not in solution.max_objectives:
                bottleneck_machines_list.append(m)
            else:
                other_machines_list.append(m)
        
        if len(other_machines_list) == 0: # Case where the objective is not aggregated by max (no bottleneck machines)
            bottleneck_machines_list.sort(key=lambda m: -1 * solution.machines[m].objective_value)
            half_index=  len(bottleneck_machines_list) // 2
            # The second half with machines that have the largest objective values goes into bottleneck 
            other_machines_list = bottleneck_machines_list[0:half_index]
            bottleneck_machines_list = bottleneck_machines_list[half_index:]

        return bottleneck_machines_list, other_machines_list
    
    
class NeighbourhoodGeneration():

    @staticmethod
    def random_swap(solution: ParallelSolution, force_improve: bool = True, internal: bool = False):
        """Performs a random swap between 2 jobs on the same machine or on different machines

        Args:
            solution (ParallelSolution): Solution to be improved
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.
            internal (bool, optional): If true, applies the swap between jobs on the same machine only. Defaults to False.

        Returns:
            ParallelSolution: New solution
        """
        # Get compatible machines (len(job_schedule) >= 1)
        compatible_machines = []
        for m in range(solution.instance.m):
            if (len(solution.machines[m].job_schedule) >= 1 and not internal) or \
                    (len(solution.machines[m].job_schedule) >= 2 and internal):
                compatible_machines.append(m)

        if len(compatible_machines) >= 2:

            random_machine_index = random.choice(compatible_machines)
            if internal:
                other_machine_index = random_machine_index
            else:
                other_machine_index = random.choice(compatible_machines)
                while other_machine_index == random_machine_index:
                    other_machine_index = random.choice(compatible_machines)

            random_machine = solution.machines[random_machine_index]
            other_machine = solution.machines[other_machine_index]

            random_machine_schedule = random_machine.job_schedule
            other_machine_schedule = other_machine.job_schedule
            
            old_obj_i, old_obj_l = random_machine.objective_value, other_machine.objective_value

            random_job_index = random.randrange(len(random_machine_schedule))
            other_job_index = random.randrange(len(other_machine_schedule))

            if internal:  # Internal swap
                while other_job_index == random_job_index:
                    other_job_index = random.randrange(
                        len(other_machine_schedule))

                new_obj_i = random_machine.simulate_swap(random_job_index, other_job_index, solution.instance)
                new_obj_l = new_obj_i
            else:  # External swap
                job_random, _, _ = random_machine_schedule[random_job_index]
                other_job, _, _ = other_machine_schedule[other_job_index]

                new_obj_i = random_machine.simulate_remove_insert(
                    random_job_index, other_job, random_job_index, solution.instance)
                new_obj_l = other_machine.simulate_remove_insert(
                    other_job_index, job_random, other_job_index, solution.instance)

            # Apply the move
            if not force_improve or (new_obj_i + new_obj_l <= old_obj_i + old_obj_l):
                random_machine_schedule[random_job_index], other_machine_schedule[
                    other_job_index] = other_machine_schedule[
                        other_job_index], random_machine_schedule[random_job_index]
                random_machine.compute_objective(solution.instance, startIndex=0)
                other_machine.compute_objective(solution.instance, startIndex=0)
                
                solution.fix_objective()

        return solution

    @staticmethod
    def random_inter_machine_insertion(solution: ParallelSolution, force_improve: bool = True):
        """Removes randomly a job from a machine and insert it on the same machine in different possition or another machine

        Args:
            solution (ParallelSolution): Solution to be improved
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.

        Returns:
            ParallelSolution: New solution
        """
        # Get compatible machines (len(job_schedule) >= 2)
        compatible_machines = []
        for m in range(solution.instance.m):
            if (len(solution.machines[m].job_schedule) >= 2):
                compatible_machines.append(m)

        if len(compatible_machines) >= 1:

            random_machine_index = random.choice(compatible_machines)
            other_mahcine_index = random.randrange(solution.instance.m)
            while other_mahcine_index == random_machine_index:
                other_mahcine_index = random.randrange(solution.instance.m)

            random_machine = solution.machines[random_machine_index]
            other_machine = solution.machines[other_mahcine_index]

            random_machine_schedule = random_machine.job_schedule
            other_machine_schedule = other_machine.job_schedule

            random_job_index = random.randrange(len(random_machine_schedule))
            other_job_index = random.randrange(len(other_machine_schedule)) if len(
                other_machine_schedule) > 0 else 0

            old_obj_i, old_obj_l = random_machine.objective_value, other_machine.objective_value
            job_i, _, _ = random_machine_schedule[random_job_index]

            new_ci = random_machine.simulate_remove_insert(random_job_index, -1, -1, solution.instance)
            new_cl = other_machine.simulate_remove_insert(-1, job_i, other_job_index, solution.instance)

            # Apply the move
            if not force_improve or (new_ci + new_cl <= old_obj_i + old_obj_l):
                job_i = random_machine_schedule.pop(random_job_index)
                other_machine_schedule.insert(other_job_index, job_i)

                random_machine.compute_objective(solution.instance, startIndex=0)
                other_machine.compute_objective(solution.instance, startIndex=0)
                solution.fix_objective()

        return solution

    @staticmethod
    def restricted_swap(solution: ParallelSolution):
        """Performs a random swap between 2 jobs of 2 different machines whose completion time is equal
        to the maximal completion time. If it's not possible, performs the move between a job on
        the machine whose completion time is equel to the maximal completion time and another
        one

        Args:
            solution (ParallelSolution): Solution to be improved

        Returns:
            ParallelSolution: New solution
        """
        bottleneck_machines_list, other_machines_list = \
                PM_LocalSearch.get_bottleneck_machines(solution)
        other_machines_list = list(filter(lambda m: len(solution.machines[m].job_schedule) >= 1 ,other_machines_list))
    
        if len(bottleneck_machines_list) > 2:
            choices = random.sample(bottleneck_machines_list, 2)
            m1, m2 = choices[0], choices[1]
        elif len(bottleneck_machines_list) == 2:
            m1, m2 = bottleneck_machines_list[0], bottleneck_machines_list[1]
        else:
            m1 = bottleneck_machines_list[0]
            if len(other_machines_list) > 0:
                m2 = random.choice(other_machines_list)
            else:
                return solution

        t1 = random.randrange(len(solution.machines[m1].job_schedule))
        t2 = random.randrange(len(solution.machines[m2].job_schedule))

        machine_1_schedule = solution.machines[m1].job_schedule
        machine_2_schedule = solution.machines[m2].job_schedule

        machine_1_schedule[t1], machine_2_schedule[t2] = machine_2_schedule[
            t2], machine_1_schedule[t1]

        solution.machines[m1].compute_objective(solution.instance, startIndex=0)
        solution.machines[m2].compute_objective(solution.instance, startIndex=0)
        solution.fix_objective()
        return solution

    @staticmethod
    def restricted_insert(solution: ParallelSolution):
        """Performs a random inter_machine_insertion between 2 different machines whose
        completion time is equal to the maximal completion time. If it's not possible, performs the
        move between a job on the machine whose completion time is equel to the 
        maximal completion time and another one

        Args:
            solution (ParallelSolution): Solution to be improved

        Returns:
            ParallelSolution: New solution
        """
        bottleneck_machines_list, other_machines_list = \
                PM_LocalSearch.get_bottleneck_machines(solution)
        
        if len(bottleneck_machines_list) > 2:
            choices = random.sample(bottleneck_machines_list, 2)
            m1, m2 = choices[0], choices[1]
        elif len(bottleneck_machines_list) == 2:
            m1, m2 = bottleneck_machines_list[0], bottleneck_machines_list[1]
        else:
            m1 = bottleneck_machines_list[0]
            m2 = random.choice(other_machines_list)

        t1 = random.randrange(len(solution.machines[m1].job_schedule))
        t2 = random.randrange(len(solution.machines[m2].job_schedule)) if len(
            solution.machines[m2].job_schedule) > 0 else 0

        machine_1_schedule = solution.machines[m1].job_schedule
        machine_2_schedule = solution.machines[m2].job_schedule

        job_i = machine_1_schedule.pop(t1)
        machine_2_schedule.insert(t2, job_i)

        solution.machines[m1].compute_objective(solution.instance, startIndex=0)
        solution.machines[m2].compute_objective(solution.instance, startIndex=0)
        solution.fix_objective()

        return solution

    @staticmethod
    def lahc_neighbour(solution_i):
        """Generates a neighbour solution of the given solution for the lahc metaheuristic

        Args:
            solution_i (ParallelSolution): Solution to be improved

        Returns:
            ParallelSolution: New solution
        """
        solution = solution_i.copy()
       
        r = random.random()
        if r < 0.5:
            solution = NeighbourhoodGeneration.random_swap(
                solution, force_improve=False, internal=False)
        else:
            solution = NeighbourhoodGeneration.random_swap(
                solution, force_improve=False, internal=True)
       
        return solution

    @staticmethod
    def SA_neighbour(solution: ParallelSolution):
        """Generates a neighbour solution of the given solution for the SA metaheuristic

        Args:
            solution_i (ParallelSolution): Solution to be improved

        Returns:
            ParallelSolution: New solution
        """
        solution_copy = solution.copy()
        r = random.random()
        if r < 0.33:
            solution_copy = NeighbourhoodGeneration.random_swap(
                solution_copy, force_improve=False, internal=False)  # External Swap
        elif r < 0.67:
            solution_copy = NeighbourhoodGeneration.random_swap(solution_copy, force_improve=False,
                                                                internal=True)  # Internal Swap
        else:
            solution_copy = NeighbourhoodGeneration.random_inter_machine_insertion(
                solution_copy, force_improve=False)  # Inter Machine Insertion
        return solution_copy

    @staticmethod
    def RSA_neighbour(solution: ParallelInstance, q0: float):
        """Generates a neighbour solution of the given solution for the lahc metaheuristic

        Args:
            solution_i (ParallelSolution): Solution to be improved
            q0 (float): Probability to apply restricted swap compared to
            restricted insertion.

        Returns:
            ParallelSolution: New solution
        """
        solution_copy = solution.copy()
        r = random.random()
        if r < q0:
            solution_copy = NeighbourhoodGeneration.restricted_swap(solution_copy)
        else:
            solution_copy = NeighbourhoodGeneration.restricted_insert(solution_copy)
        return solution_copy