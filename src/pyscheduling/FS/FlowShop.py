import json
import random
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np

import pyscheduling.Problem as Problem
from pyscheduling.Problem import Job
from pyscheduling.BaseConstraints import BaseConstraints as Constraints


class GenerationProtocol(Enum):
    BASE = 1

@dataclass
class FlowShopInstance(Problem.BaseInstance):

    n: int  # n : Number of jobs
    m: int  # m : Number of machines

    def __init__(self, n: int, m: int, name: str = "Unknown", **kwargs):
        super().__init__(n, m=m, name = name, **kwargs)


@dataclass
class Machine:
    
    objective_value: int = 0
    last_job: int = -1
    oper_schedule: List[Job] = field(default_factory=list)
    
    def __init__(self, machine_num:int, oper_schedule: List[Job] = None, last_job: int = -1, objective_value: int = 0) -> None:
        """Constructor of Machine

        Args:
            objective (int, optional): completion time of the last job of the machine. Defaults to 0.
            last_job (int, optional): ID of the last job set on the machine. Defaults to -1.
            job_schedule (list[Job], optional): list of Jobs scheduled on the machine in the exact given sequence. Defaults to None.
        """
        self.machine_num = machine_num
        self.objective_value = objective_value
        self.last_job = last_job
        if oper_schedule is None:
            self.oper_schedule = []
        else:
            self.oper_schedule = oper_schedule

    def __str__(self):
        return " : ".join(map(str, [(job.id, job.start_time, job.end_time) for job in self.oper_schedule])) + " | " + str(self.objective_value)

    def __eq__(self, other):
        same_schedule = other.oper_schedule == self.oper_schedule
        return (same_schedule)

    def copy(self):
        return Machine(self.machine_num, list(self.oper_schedule), self.last_job, self.objective_value)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    @staticmethod
    def fromDict(machine_dict):
        return Machine(machine_dict["objective"], machine_dict["last_job"], machine_dict["job_schedule"])

    def compute_current_ci(self, instance: FlowShopInstance, prev_machine_ci: int ,prev_job_ci: int, job_prev_i: int, job_i: int):
        """Computes the current ci when job_i comes after job_prev_i.
        This takes into account if we have setup times and release dates.

        Args:
            instance (SingelInstance): the instance to be solved.
            prev_machine_ci (int): the ci of the same job on the previous machine
            prev_ci (int): the completion time of the previous job on the same machine
            job_prev_i (int): id of the job that precedes the inserted job 
            job_i (int): id of the job to be inserted at the end

        Returns:
            tuple: (ci, start_time), the new completion time and start_time of the inserted job.
        """
        startTime = max(prev_machine_ci, prev_job_ci, instance.R[job_i]) if hasattr(instance, 'R') else max(prev_machine_ci, prev_job_ci)
        setupTime = instance.S[self.machine_num][job_prev_i][job_i] if hasattr(instance, 'S') else 0
        remaining_setupTime = max(setupTime-(startTime-prev_job_ci),0)
        proc_time = instance.P[job_i][self.machine_num]

        ci = startTime + remaining_setupTime + proc_time
        return startTime, ci
    
    def fix_schedule(self, instance: FlowShopInstance, prev_machine, startIndex: int = 0):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job

        Args:
            instance (SingleInstance): The instance associated to the machine
            startIndex (int) : The job index the function starts operating from

        Returns:
            int: objective
        """
        if startIndex > 0:
            job_prev_i, ci = self.oper_schedule[startIndex - 1].id, self.oper_schedule[startIndex - 1].end_time
        else:
            job_prev_i, ci = (self.oper_schedule[startIndex].id, 0)
        
        for i in range(startIndex, len(self.oper_schedule)):
            job_i = self.oper_schedule[i].id
            prev_machine_ci = prev_machine.oper_schedule[i].end_time if prev_machine is not None else 0
            start_time, ci = self.compute_current_ci(instance, prev_machine_ci, ci, job_prev_i, job_i)
            self.oper_schedule[i] = Job(job_i, start_time, ci)
            job_prev_i = job_i

        self.last_job = job_i
        self.objective_value = ci
        return ci

    def idle_time(self):
        """returns the idle time on the machine

        Returns:
            int: idle time of the machine
        """
        idleTime = self.oper_schedule[0].start_time
        for job_index in range(len(self.oper_schedule)-1):
            idleTime += self.oper_schedule[job_index+1].start_time - self.oper_schedule[job_index].end_time
        return idleTime

@dataclass
class FlowShopSolution(Problem.BaseSolution):

    machines: List[Machine]
    job_schedule = List[int]

    def __init__(self, instance: FlowShopInstance = None, machines: List[Machine] = None, job_schedule : List[int] = None, objective_value: int = 0):
        """Constructor of RmSijkCmax_Solution

        Args:
            instance (FlowShopInstance, optional): Instance to be solved by the solution. Defaults to None.
            machines (list[Machine], optional): list of machines of the instance. Defaults to None.
            job_schedule (list[int], optional): sequence of scheduled jobs. Defaults to None.
            objective_value (int, optional): initial objective value of the solution. Defaults to 0.
        """
        self.instance = instance
        if machines is None:
            self.machines = []
            for i in range(instance.m):
                machine = Machine(i, [], -1, 0)
                self.machines.append(machine)
        else:
            self.machines = machines
        if job_schedule is None: 
            self.job_schedule = []
        else: 
            self.job_schedule = job_schedule
        self.objective_value = objective_value

    def __str__(self):
        return "Objective : " + str(self.objective_value) + "\n" + "Jobs sequence : " + "\t".join(map(str, self.job_schedule)) + "\n" + "Machine_ID | Job_schedule (job_id , start_time , completion_time) | Completion_time\n" + "\n".join(map(str, self.machines))

    def copy(self):
        copy_machines = []
        for m in self.machines:
            copy_machines.append(m.copy())

        copy_solution = FlowShopSolution(self.instance, copy_machines,
                                        list(self.job_schedule), self.objective_value)
        return copy_solution

    def __lt__(self, other):
        if self.instance.get_objective().value > 0 :
            return self.objective_value < other.objective_value
        else : return other.objective_value < self.objective_value
    
    def propagate_schedule(self, startIndex: int=0):
        """Fills the job_schedule of every machine from job_schedule of Solution
        """
        if startIndex == 0: 
            for machine in self.machines :
                machine.oper_schedule = [Job(job.id,0,0) for job in self.job_schedule]

        else:
            if len(self.job_schedule) != len(self.machines[0].oper_schedule): # Add the missing jobs or remove the surplus
                diff = len(self.job_schedule) - len(self.machines[0].oper_schedule)
                if diff > 0: # There are missing jobs in machines
                    for machine in self.machines :
                        machine.oper_schedule.extend( [Job(self.instance.n + 1, 0, 0) for _ in range(diff) ] )
                else: # There are more jobs in machines, delete the last diff jobs
                    for machine in self.machines :
                        del machine.oper_schedule[diff:]

            # Change jobs order from startIndex
            for machine in self.machines:
                for i in range(startIndex, len(self.job_schedule)):
                    job_id, start, end = self.job_schedule[i]
                    machine.oper_schedule[i] = Job(job_id, start, end)

    def fix_objective(self):
        objective = self.instance.get_objective()
        if objective == Problem.Objective.Cmax:
            self.objective_value = self.machines[-1].objective_value
        elif objective == Problem.Objective.wiCi:
            self.objective_value = sum( self.instance.W[job.id] * job.end_time for job in self.machines[-1].oper_schedule )
        elif objective == Problem.Objective.wiFi:
            self.objective_value = sum( self.instance.W[job.id] * (job.end_time - self.instance.R[job.id]) for job in self.machines[-1].oper_schedule )
        elif objective == Problem.Objective.wiTi:
            self.objective_value = sum( self.instance.W[job.id] * max(job.end_time-self.instance.D[job.id],0) for job in self.machines[-1].oper_schedule )
        elif objective == Problem.Objective.Lmax:
            self.objective_value = max( 0, max( job.end_time-self.instance.D[job.id] for job in self.machines[-1].oper_schedule ) )

        return self.objective_value

    def compute_objective(self, startIndex: int = 0):
        self.propagate_schedule(startIndex)
        prev_machine = None
        for i, machine in enumerate(self.machines):
            machine.fix_schedule(self.instance, prev_machine, startIndex)
            prev_machine = machine
        
        for j in range(len(self.job_schedule)):
            job_id = self.job_schedule[j].id
            start_time = self.machines[0].oper_schedule[j].start_time
            end_time = self.machines[-1].oper_schedule[j].end_time
            self.job_schedule[j] = Job( job_id, start_time, end_time )

        return self.fix_objective()

    def simulate_insert_last(self, job_id : int):
        """returns start_time and completion_time of job_id if scheduled at the end of job_schedule
        at every stage (machine)

        Args:
            job_id (int): job to be scheduled at the end

        Returns:
            int, int: start_time of job_id, completion_time of job_id
        """ 
        prev_machine_ci = 0
        for i, machine in enumerate(self.machines):
            prev_job, _, prev_ci = machine.oper_schedule[-1] if len(machine.oper_schedule) > 0 else (job_id, 0, 0)
            start_time, end_time = machine.compute_current_ci(self.instance, prev_machine_ci, prev_ci, prev_job, job_id)
            prev_machine_ci = end_time

        return start_time, end_time
    
    def simulate_insert_objective(self, job_id, start_time, end_time):
        """Returns the new objective if job_id is inserted at the end with start_time and end_time

        Args:
            job_id (int): id of the inserted job
            start_time (int): start time of the job
            end_time (int): end time of the job
        
        Returns:
            int: the new objective
        """
        objective = self.instance.get_objective()
        if objective == Problem.Objective.Cmax:
            return end_time
        elif objective == Problem.Objective.wiCi:
            return self.objective_value + self.instance.W[job_id] * end_time
        elif objective == Problem.Objective.wiFi:
            return self.objective_value + self.instance.W[job_id] * (end_time - self.instance.R[job_id])
        elif objective == Problem.Objective.wiTi:
            return self.objective_value + self.instance.W[job_id] * max(end_time-self.instance.D[job_id],0)
        elif objective == Problem.Objective.Lmax:
            return max( self.objective_value, end_time-self.instance.D[job_id]) 

    def idle_time(self):
        """returns the idle time of the last machine

        Returns:
            int: idle time of the last machine
        """
        return self.machines[-1].idle_time()
    
    @classmethod
    def read_txt(cls, path: Path):
        """Read a solution from a txt file

        Args:
            path (Path): path to the solution's txt file of type Path from pathlib

        Returns:
            FlowShopSolution:
        """
        f = open(path, "r")
        content = f.read().split('\n')
        objective_value_ = int(content[0].split(':')[1])
        configuration_ = []
        for i in range(2, len(content)):
            line_content = content[i].split('|')
            configuration_.append(Machine(int(line_content[0]), int(line_content[2]), oper_schedule=[Job(
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

    def plot(self, types: str = "RSP", path: Path = None):
        """Plot the solution into a gantt diagramm

        Args:
            types (str, optional): The type of tasks to plot, a string where each letter corresponds to one type: R for release, \n
                S for setup and P for Processing. Defaults to "RSP".
            path (Path, optional): The path to export the diagram, if not specified it is not exported but shown inline. Defaults to None.
        """
        types_upper = types.upper()
        plot_proc = "P" in types_upper
        plot_setup = "S" in types_upper
        plot_release = "R" in types_upper
        tasks_df = []
        cmax_value = 0
        for j in range(len(self.machines)):
            schedule = self.machines[j].oper_schedule
            prev = schedule[0].id # First job in the schedule
            prevEndTime = 0
            for element in schedule:
                job_index, startTime, endTime = element
                
                # Idle time
                if prevEndTime < startTime and plot_release:
                    idle_time_task = dict(Task=f"M{j}", Description=f"J{job_index} idle", Start=prevEndTime,
                                        Finish = startTime, Type = 'Idle')
                    tasks_df.append(idle_time_task)
                
                # Setup time
                end_setup = startTime + self.instance.S[j][prev][job_index] if hasattr(self.instance, "S") else startTime
                if startTime != end_setup and plot_setup:
                    setup_task = dict(Task=f"M{j}", Description=f"J{job_index} setup",
                                    Start = startTime, Finish = end_setup, Type = 'Setup')
                    tasks_df.append(setup_task)
                
                # Proc time
                end_proc = end_setup + self.instance.P[job_index][j]
                if plot_proc:
                    proc_task = dict(Task=f"M{j}", Description=f"J{job_index} proc", 
                                    Start = end_setup, Finish = end_proc, Type = 'Processing')
                    tasks_df.append(proc_task)
                
                prev = job_index
                prevEndTime = endTime
                cmax_value = max(cmax_value, endTime)

        self._plot_tasks(tasks_df, path)

    def is_valid(self):
        """
        Check if solution respects the constraints
        """
        is_valid = True
        set_jobs = set( job.id for job in self.job_schedule )
        if len(set_jobs) != self.instance.n or len(self.job_schedule) != self.instance.n:
            print("The set of scheduled jobs in solution is not the one expected, missing or extra jobs are found!")
            is_valid = False
        job_times = {i: (0, 0) for i in range(self.instance.n)}

        for k, machine in enumerate(self.machines):
            
            if len(machine.oper_schedule) != len(self.job_schedule): # Same length
                print(f"There is a difference between the set of jobs " +
                    f"in solution ({len(self.job_schedule)}) and in machine ({len(machine.oper_schedule)})")

            expected_start_time, setup_time, ci  = 0, 0, 0
            prev_job = None
            for i, job_element in enumerate(machine.oper_schedule):
                job_id, startTime, endTime = job_element

                if i < len(self.job_schedule) and job_id != self.job_schedule[i].id: # Same jobs
                    print(f"Difference between job schdule in solution and "+
                    f"machine {k} oper schedule in position {i}: solution[{i}] = {self.job_schedule[i].id}"+
                    f" machine[{i}] = {job_id}")
                
                # Check start and end times
                prev_start, prev_end = job_times[job_id]
                if hasattr(self.instance,'R'):
                    expected_start_time = max(self.instance.R[job_id], prev_end, ci)
                else: 
                    expected_start_time = max(prev_end, ci)

                # Check setup constraint
                setup_time = 0
                if hasattr(self.instance,'S'):
                    if prev_job is None:
                        setup_time = self.instance.S[machine.machine_num][job_id][job_id]
                    else:
                        setup_time = self.instance.S[machine.machine_num][prev_job][job_id]
                
                remaining_setupTime = max(setup_time-(expected_start_time-ci),0)
                proc_time = self.instance.P[job_id][machine.machine_num]
                ci = expected_start_time + remaining_setupTime + proc_time         

                if startTime != expected_start_time or endTime != ci:
                    print(f'## Error: in machine {machine.machine_num}' +
                          f' found {job_element} expected {job_id,expected_start_time, ci}')
                    is_valid = False
                
                start_job = expected_start_time if k == 0 else prev_start
                job_times[job_id] = (start_job, ci)

                prev_job = job_id
            
        objective = self.instance.get_objective()
        expected_obj = 0
        if objective == Problem.Objective.Cmax:
            expected_obj = max(period[1] for period in job_times.values())
        elif objective == Problem.Objective.wiCi:
            expected_obj = sum( self.instance.W[i] * job_times[i][1] for i in job_times )
        elif objective == Problem.Objective.wiFi:
            expected_obj = sum( self.instance.W[i] * (job_times[i][1] - self.instance.R[i]) for i in job_times)
        elif objective == Problem.Objective.wiTi:
            expected_obj = sum( self.instance.W[i] * max(job_times[i][1]-self.instance.D[i],0) for i in job_times )

        if expected_obj != self.objective_value:
            print(f'## Error: in solution' +
                    f' found objective_value = {self.objective_value} expected {expected_obj}')
            is_valid = False

        return is_valid


class FS_LocalSearch(Problem.LocalSearch):
    
    @staticmethod
    def _iterative_best_insert(solution: FlowShopSolution, inplace: bool = True):
        
        solution_copy = solution.copy() if not inplace else solution
        jobs_list = [job.id for job in solution_copy.job_schedule]
        for job_id in jobs_list:
            for i, job in enumerate(solution_copy.job_schedule):
                if job.id == job_id:
                    pos = i
                    break
            
            job = solution_copy.job_schedule[pos]
            old_objective = solution_copy.objective_value
            taken_pos = pos
            prev_pos = pos
            for new_pos in range(len(solution_copy.job_schedule)):
                if(pos != new_pos):
                    # Apply the insertion
                    taken_job = solution_copy.job_schedule.pop(prev_pos)
                    solution_copy.job_schedule.insert(new_pos, taken_job)
                    new_objective = solution_copy.compute_objective(min(prev_pos, new_pos))

                    if new_objective < old_objective:
                        taken_pos = new_pos
                        old_objective = new_objective
                    prev_pos = new_pos
            if taken_pos != prev_pos:
                solution_copy.job_schedule.pop(prev_pos)
                solution_copy.job_schedule.insert(taken_pos, job)
                solution_copy.compute_objective(min(taken_pos, prev_pos))
        return solution_copy
    
    @staticmethod
    def _iterative_best_swap(solution: FlowShopSolution, inplace: bool = True):

        solution_copy = solution.copy() if not inplace else solution
        job_schedule_len = len(solution_copy.job_schedule)
        old_obj = solution_copy.objective_value
        for i in range(0, job_schedule_len):
            move = None
            for j in range(i+1, job_schedule_len):

                solution_copy.job_schedule[i], solution_copy.job_schedule[j] = solution_copy.job_schedule[j], solution_copy.job_schedule[i]
                new_objective = solution_copy.compute_objective(startIndex= min(i, j))
                solution_copy.job_schedule[i], solution_copy.job_schedule[j] = solution_copy.job_schedule[j], solution_copy.job_schedule[i]

                if new_objective < old_obj and (move is None or new_objective < move[2]):
                    move = (i, j, new_objective)

            if not move is None:
                i, j, new_objective = move
                solution_copy.job_schedule[i], solution_copy.job_schedule[j] = solution_copy.job_schedule[j], solution_copy.job_schedule[i]
                solution_copy.compute_objective(startIndex= min(i, j))
                old_obj = solution_copy.objective_value
            else:
                solution_copy.compute_objective()

        return solution_copy

class NeighbourhoodGeneration():

    @staticmethod
    def random_insert(solution: FlowShopSolution, force_improve: bool = False, inplace: bool = False, nb_moves: int = 1):
        """Performs an insert of a random job in a random position

        Args:
            solution (FlowShopSolution): Solution to be improved
            objective (RootProblem.Objective) : objective to consider
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.

        Returns:
            FlowShopSolution: New solution
        """
        if not inplace or force_improve:
            solution_copy = solution.copy()
        else:
            solution_copy = solution

        job_schedule = solution_copy.job_schedule
        job_schedule_len = len(job_schedule)
        old_objective = solution_copy.objective_value

        for _ in range(nb_moves):
            # Get the job and the position
            random_job_index = random.randrange(job_schedule_len)
            random_pos = random_job_index
            while random_pos == random_job_index:
                random_pos = random.randrange(job_schedule_len)

            # Simulate applying the insertion move
            random_job = job_schedule.pop(random_job_index)
            job_schedule.insert(random_pos, random_job)
            new_objective = solution_copy.compute_objective(startIndex= min(random_job_index, random_pos))

        # Update the solution
        if force_improve and (new_objective > old_objective):
            return solution

        return solution_copy

    @staticmethod
    def random_swap(solution: FlowShopSolution, force_improve: bool = False, inplace: bool = False, nb_moves: int = 1):
        """Performs a random swap between 2 jobs

        Args:
            solution (FlowShopSolution): Solution to be improved
            objective (RootProblem.Objective) : objective to consider
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.

        Returns:
            FlowShopSolution: New solution
        """
        if not inplace or force_improve:
            solution_copy = solution.copy()
        else:
            solution_copy = solution

        # Select the two different random jobs to be swapped 
        job_schedule = solution_copy.job_schedule
        job_schedule_len = len(job_schedule)
        old_objective = solution_copy.objective_value

        for _ in range(nb_moves):

            random_job_index = random.randrange(job_schedule_len)
            other_job_index = random.randrange(job_schedule_len)
            while other_job_index == random_job_index:
                other_job_index = random.randrange(job_schedule_len)

            # Simulate applying the swap move
            job_schedule[random_job_index], job_schedule[other_job_index] = job_schedule[
                        other_job_index], job_schedule[random_job_index]
            
            new_objective = solution_copy.compute_objective(startIndex= min(random_job_index, other_job_index))

        # Update the solution
        if force_improve and (new_objective > old_objective):
            return solution

        return solution_copy
    
    @staticmethod
    def random_neighbour(solution_i: FlowShopSolution):
        """Generates a random neighbour solution of the given solution

        Args:
            solution_i (FlowShopSolution): Solution at iteration i

        Returns:
            FlowShopSolution: New solution
        """ 
        r = random.random()
        if r < 0.5:
            solution = NeighbourhoodGeneration.random_insert(
                solution_i, force_improve=False, inplace=False, nb_moves=2)
        else:
            solution = NeighbourhoodGeneration.random_swap(
                solution_i, force_improve=False, inplace=False, nb_moves=2)
       
        return solution

    @staticmethod
    def deconstruct_construct(solution_i: FlowShopSolution, d: float = 0.25):
        """Generates a random neighbour solution of the given solution using the deconstruct - construct strategy

        The procedure removes a set of jobs and insert them using best insertion (greedy) 

        Args:
            solution_i (FlowShopSolution): Solution at iteration i

        Returns:
            FlowShopSolution: New solution
        """ 
        solution_copy = solution_i.copy()
        # Deconstruction of d (percentage) random jobs out all jobs
        all_jobs = list(range(solution_copy.instance.n))
        nb_removed_jobs = int(solution_copy.instance.n * d )
        removed_jobs = random.sample(all_jobs, nb_removed_jobs)

        solution_copy.job_schedule = [job for job in solution_copy.job_schedule if job.id not in removed_jobs]
        solution_copy.compute_objective()

        # Construction by inserting the removed jobs one by one
        for n_j, j in enumerate(removed_jobs):
            move = None
            prev_pos = None
            for pos in range(len(solution_copy.job_schedule) + 1):
                if prev_pos is not None:
                    solution_copy.job_schedule.pop(prev_pos)

                solution_copy.job_schedule.insert(pos, Job(j, 0, 0))
                new_obj = solution_copy.compute_objective(startIndex=pos-1)
                prev_pos = pos

                if move is None or move[1] > new_obj:
                    move = (pos, new_obj)
            
            # Apply the best move
            solution_copy.job_schedule.pop(prev_pos)
            solution_copy.job_schedule.insert(move[0], Job(j, 0, 0))
            solution_copy.compute_objective(startIndex=move[0])

        return solution_copy