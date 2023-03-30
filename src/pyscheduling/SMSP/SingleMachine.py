import itertools
import json
import random
import sys
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

from pyscheduling.Problem import Job, Objective, RandomDistrib, GenerationProtocol
import pyscheduling.Problem as Problem
from pyscheduling.SMSP.Constraints import Constraints

class GenerationProtocol(Enum):
    BASE = 1

@dataclass
class SingleInstance(Problem.BaseInstance):

    n: int

    def __init__(self, n: int, name: str = "Unknown", **kwargs):
        super().__init__(n, name = name, **kwargs)

    def generate_P(self, protocol: GenerationProtocol, law: RandomDistrib, Pmin: int, Pmax: int):
        """Random generation of processing time table

        Args:
            protocol (GenerationProtocol): given protocol of generation of random instances
            law (GenerationLaw): probablistic law of generation
            Pmin (int): Minimal processing time
            Pmax (int): Maximal processing time

        Returns:
           list[int]: Table of processing time
        """
        P = []
        for j in range(self.n):
            if law.name == "UNIFORM":  # Generate uniformly
                n = int(random.uniform(Pmin, Pmax))
            elif law.name == "NORMAL":  # Use normal law
                value = np.random.normal(0, 1)
                n = int(abs(Pmin+Pmax*value))
                while n < Pmin or n > Pmax:
                    value = np.random.normal(0, 1)
                    n = int(abs(Pmin+Pmax*value))
            P.append(n)

        return P


@dataclass
class Machine:

    objective_value: int = 0
    last_job: int = -1
    job_schedule: List[Job] = field(default_factory=list)
    # this table serves as a cache to save the total weighted completion time reached after each job in job_schedule
    wiCi_cache: List[int] = field(default_factory=list)
    # this table serves as a cache to save the total weighted lateness reached after each job in job_schedule
    wiTi_cache: List[int] = field(default_factory=list)
    wiFi_cache: List[int] = field(default_factory=list)

    def __init__(self, objective_value: int = 0, last_job: int = -1, job_schedule: List[Job] = None, wiCi_cache: List[int] = None, wiTi_cache: List[int] = None, wiFi_cache: List[int] = None):
        """Constructor of Machine

        Args:
            objective (int, optional): completion time of the last job of the machine. Defaults to 0.
            last_job (int, optional): ID of the last job set on the machine. Defaults to -1.
            job_schedule (list[Job], optional): list of Jobs scheduled on the machine in the exact given sequence. Defaults to None.
        """
        self.objective_value = objective_value
        self.last_job = last_job
        if job_schedule is None:
            self.job_schedule = []
        else:
            self.job_schedule = job_schedule
        self.wiCi_cache = wiCi_cache
        self.wiTi_cache = wiTi_cache
        self.wiFi_cache = wiFi_cache
        self.objectives_map = {
            Objective.wiCi: self.wiCi_cache,
            Objective.wiTi: self.wiTi_cache,
            Objective.wiFi: self.wiFi_cache
        }
      
    def __repr__(self):
        return f"Machine 0" + " -> ".join(map(str, [(job.id, job.start_time, job.end_time) for job in self.job_schedule])) + " | Objective value: " + str(self.objective_value)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        same_schedule = other.job_schedule == self.job_schedule
        return (same_schedule)

    def copy(self):
        wiCi_copy = self.wiCi_cache if self.wiCi_cache is None else list(self.wiCi_cache)
        wiTi_copy = self.wiTi_cache if self.wiTi_cache is None else list(self.wiTi_cache)
        wiFi_copy = self.wiFi_cache if self.wiFi_cache is None else list(self.wiFi_cache)
        return Machine(self.objective_value, self.last_job, list(self.job_schedule),
                        wiCi_cache=wiCi_copy, wiTi_cache=wiTi_copy, wiFi_cache=wiFi_copy)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    @staticmethod
    def fromDict(machine_dict: dict):
        return Machine(machine_dict["objective"], machine_dict["last_job"], machine_dict["job_schedule"])

    def compute_current_ci(self, instance: SingleInstance, prev_ci: int, job_prev_i: int, job_i: int):
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
        setupTime = instance.S[job_prev_i][job_i] if hasattr(instance, 'S') and job_prev_i != -1 else 0
        proc_time = instance.P[job_i]

        ci = startTime + setupTime + proc_time
        return ci, startTime

    def init_cache(self, instance: SingleInstance, startIndex: int = 0):
        """Initialize the cache if it's not defined yet

        Args:
            startIndex (int, optional): The index from which we start fixing the schedule. Defaults to 0.
            obj_cache (list[int]): The objective's cache, it can be wiCi, wiTi or other. Defaults to None.

        Returns:
            tuple: (startIndex, obj_cache) 
        """
        objective = instance.get_objective()
        obj_cache = self.objectives_map.get(objective, -1)
        if obj_cache == -1: # No cache is used
            return startIndex, None
        if obj_cache is None:  # Initialize obj_cache to the size of job_schedule
            obj_cache = [-1] * len(self.job_schedule)
            self.objectives_map[objective] = obj_cache
            startIndex = 0
            
        elif len(obj_cache) != len(self.job_schedule):
            obj_cache.insert(startIndex, -1) # Insert an element in obj_cache corresponding to the position where a new job has been inserted
        
        return startIndex, obj_cache
    
    def init_obj(self, startIndex: int = 0, obj_cache: List[int] = None):
        """This is a helper method to initialize the values of ci, prev_job and objective from the current schedule and the objective cache if present.

        Args:
            startIndex (int, optional): The index from which we start fixing the schedule. Defaults to 0.
            obj_cache (list[int], optional): The objective's cache, it can be wiCi, wiTi or other. Defaults to None.

        Returns:
            tuple: ci, job_prev_i, obj
        """
        if startIndex > 0:
            ci = self.job_schedule[startIndex - 1].end_time
            obj = obj_cache[startIndex - 1] if obj_cache is not None else 0 # When the cache is not specified, the objective does not depend on it (like Cmax)
            job_prev_i = self.job_schedule[startIndex - 1].id
        else:
            if len(self.job_schedule) >0:
                ci, obj, job_prev_i,  = 0, 0, self.job_schedule[startIndex].id
            else:
                ci, obj, job_prev_i = 0,0,-1
            
        return ci, job_prev_i, obj
    
    def compute_obj_from_ci(self, instance: SingleInstance, ci: int, job_i: int, curr_obj: int):
        """Helper method to compute the objective value from the current ci.
        According to the objective set on the instance, the expression of the objective in function of ci changes 

        Args:
            instance (SingleInstance): the current problem instance
            ci (int): the current completion time
            job_i (int): the job that was inserted
            curr_obj (int): current objective before inserting the job (cumulative)

        Returns:
            int: obj, the new objective
        """
        objective = instance.get_objective()
        if objective == Objective.Cmax:
            return ci
        elif objective == Objective.wiCi:
            return curr_obj + instance.W[job_i] * ci
        elif objective == Objective.wiTi:
            return curr_obj + instance.W[job_i] * max(ci-instance.D[job_i], 0)
        elif objective == Objective.wiFi:
            return curr_obj + instance.W[job_i] * (ci-instance.R[job_i])
        elif objective == Objective.Lmax:
            return max(curr_obj, ci - instance.D[job_i])

    def compute_objective(self, instance: SingleInstance, startIndex: int = 0):
        """Fills the job_schedule with the correct sequence of start_time and completion_time of each job and returns the objective

        Args:
            instance (SingleInstance): The instance associated to the machine
            startIndex (int) : The job index the function starts operating from

        Returns:
            int: objective
        """
        startIndex, obj_cache = self.init_cache(instance, startIndex)
        ci, job_prev_i, obj = self.init_obj(startIndex, obj_cache)
        for i in range(startIndex, len(self.job_schedule)):
            job_i = self.job_schedule[i].id
            ci, start_time = self.compute_current_ci(instance, ci, job_prev_i, job_i)
            self.job_schedule[i] = Job(job_i, start_time, ci)

            obj = self.compute_obj_from_ci(instance, ci, job_i, obj) # This is cumulative objective
            if obj_cache is not None:
                obj_cache[i] = obj
            job_prev_i = job_i

        self.objective_value = obj
        
        return obj

    def simulate_remove_insert(self, pos_remove: int, job: int, pos_insert: int, instance:  SingleInstance):
        """Computes the objective if we remove job at position "pos_remove" 
        and insert "job" at "pos" in the machine's job_schedule

        Args:
            pos_remove (int): position of the job to be removed
            job (int): id of the inserted job
            pos_insert (int): position where the job is inserted in the machine
            instance (SingleInstance): the current problem instance
        Returns:
            int: total weighted completion time
        """
        first_pos = min(pos_remove, pos_insert) if pos_remove != -1 and pos_insert != -1 else max(pos_remove, pos_insert)
        job_prev_i, _, ci = self.job_schedule[first_pos - 1] if first_pos > 0 else (job, 0, 0)
        job_prev_i = self.job_schedule[1].id if pos_remove == 0 and pos_insert != 0 and len(self.job_schedule) > 1 else job_prev_i # The case when we remove the first job
        
        objective = instance.get_objective()
        obj_cache = self.objectives_map.get(objective, None)
        obj = obj_cache[first_pos - 1] if first_pos > 0 and obj_cache is not None else 0
        for i in range(first_pos, len(self.job_schedule) + 1): # +1 for edge cases when inserting in empty schedule or at the very end of a non empty schedule

            # If job needs to be inserted to position i
            if i == pos_insert:
                ci, _ = self.compute_current_ci(instance, ci, job_prev_i, job)
                obj = self.compute_obj_from_ci(instance, ci, job, obj)
                job_prev_i = job # Since the inserted job now preceeds the next job

            # If the job_i is not the one to be removed
            if i != pos_remove and i < len(self.job_schedule):
                job_i = self.job_schedule[i].id
                ci, _ = self.compute_current_ci(instance, ci, job_prev_i, job_i)
                obj = self.compute_obj_from_ci(instance, ci, job_i, obj)
                job_prev_i = job_i # TODO: job_prev_i should be updated only when it is not removed

        return obj

    def simulate_swap(self, pos_i: int, pos_j: int, instance: SingleInstance):
        """Computes the objective if we insert swap jobs at position "pos_i" and "pos_j"
        in the machine's job_schedule
        Args:
            pos_i (int): position of the first job to be swapped
            pos_j (int): position of the second job to be swapped
            instance (SingleInstance): the current problem instance
        Returns:
            int : total weighted completion time
        """
        first_pos = min(pos_i, pos_j)
        first_job = self.job_schedule[first_pos].id
        job_prev_i, _, ci = self.job_schedule[first_pos - 1] if first_pos > 0 else (first_job, 0, 0)

        objective = instance.get_objective()
        obj_cache = self.objectives_map.get(objective, None)
        obj = obj_cache[first_pos - 1] if first_pos > 0 and obj_cache is not None else 0

        for i in range(first_pos, len(self.job_schedule)):

            if i == pos_i:  # We take pos_j
                job_i = self.job_schedule[pos_j].id  # (Id, startTime, endTime)
            elif i == pos_j:  # We take pos_i
                job_i = self.job_schedule[pos_i].id
            else:
                job_i = self.job_schedule[i].id  # Id of job in position i

            ci, _ = self.compute_current_ci(instance, ci, job_prev_i, job_i)
            obj = self.compute_obj_from_ci(instance, ci, job_i, obj)
            job_prev_i = job_i

        return obj

    def maximum_lateness(self, instance : SingleInstance):
        job_schedule_len = len(self.job_schedule)
        if job_schedule_len > 0 :
            ci = 0
            maximum_lateness = 0
            job_prev_i = self.job_schedule[0].id
            for i in range(0,job_schedule_len):
                job_i = self.job_schedule[i].id
                if hasattr(instance, 'R'):
                    startTime = max(ci, instance.R[job_i])
                else:
                    startTime = ci
                if hasattr(instance, 'S'):
                    setupTime = instance.S[job_prev_i][job_i]
                else:
                    setupTime = 0
                proc_time = instance.P[job_i]
                ci = startTime + setupTime + proc_time

                self.job_schedule[i] = Job(job_i, startTime, ci)
                maximum_lateness = max(maximum_lateness,ci - instance.D[job_i])
                job_prev_i = job_i
        self.objective = maximum_lateness
        return maximum_lateness


@dataclass
class SingleSolution(Problem.BaseSolution):

    machine: Machine

    def __init__(self, instance: SingleInstance, machine: Machine = None, objective_value: int = 0):
        """Constructor of SingleSolution

        Args:
            instance (SingleInstance, optional): Instance to be solved by the solution.
        """
        self.instance = instance
        self.objective = instance.get_objective()
        if machine is None:
            self.machine = Machine(0, -1, [])
        else:
            self.machine = machine
        self.objective_value = objective_value

    def __repr__(self):
        return "Objective : " + str(self.objective_value) + "\n" + "Job_schedule (job_id , start_time , completion_time) | objective\n" + self.machine.__str__()

    def __str__(self):
        return self.__repr__()
        
    def copy(self):
        copy_machine = self.machine.copy()
        copy_solution = SingleSolution(self.instance, machine=copy_machine, objective_value=self.objective_value)
        return copy_solution

    def __lt__(self, other):
        if self.instance.get_objective().value > 0:
            return self.objective_value < other.objective_value
        else:
            return other.objective_value < self.objective_value

    def fix_objective(self):
        """Sets the objective_value attribute of the solution to the objective attribute of the machine
        """
        self.objective_value = self.machine.objective_value

    def compute_objective(self):
        """Computes the current solution's objective.
            By calling the compute objective on the only existing machine and setting the returned value.
        """
        self.machine.compute_objective(self.instance)
        self.fix_objective()

    def Lmax(self):
        """Sets the job_schedule of the machine and affects the maximum lateness to the objective_value attribute
        """
        if self.instance != None:
                self.machine.maximum_lateness(self.instance)
        self.objective_value = self.machine.objective
        return self.objective_value

    @classmethod
    def read_txt(cls, path: Path):
        """Read a solution from a txt file

        Args:
            path (Path): path to the solution's txt file of type Path from pathlib

        Returns:
            SingleSolution:
        """
        f = open(path, "r")
        content = f.read().split('\n')
        objective_value_ = int(content[0].split(':')[1])
        line_content = content[2].split('|')
        machine = Machine(int(line_content[1]), job_schedule=[Job(
            int(j[0]), int(j[1]), int(j[2])) for j in [job.strip()[1:len(job.strip())-1].split(',') for job in line_content[0].split(':')]])
        solution = cls(objective_value=objective_value_,
                       machine=machine)
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
        schedule = self.machine.job_schedule
        prev = schedule[0].id # First job in the schedule
        prevEndTime = 0
        for element in schedule:
            job_index, startTime, endTime = element
            
            # Idle time
            if prevEndTime < startTime:
                idle_time_task = dict(Task=f"M0", Description=f"J{job_index} idle", Start=prevEndTime,
                                    Finish = startTime, Type = 'Idle')
                tasks_df.append(idle_time_task)
            
            # Setup time
            end_setup = startTime + self.instance.S[prev][job_index] if hasattr(self.instance, "S") else startTime
            if startTime != end_setup:
                setup_task = dict(Task=f"M0", Description=f"J{job_index} setup",
                                Start = startTime, Finish = end_setup, Type = 'Setup')
                tasks_df.append(setup_task)
            
            # Proc time
            end_proc = end_setup + self.instance.P[job_index]
            proc_task = dict(Task=f"M0", Description=f"J{job_index} proc", 
                            Start = end_setup, Finish = end_proc, Type = 'Processing')
            tasks_df.append(proc_task)
            
            prev = job_index
            prevEndTime = endTime

        return self._plot_tasks(tasks_df, path)

    def is_valid(self, verbosity : bool = False):
        """
        Check if solution respects the constraints
        """
        set_jobs = set()
        is_valid = True
        prev_job = None
        expected_start_time, setup_time, ci = 0, 0, 0
        obj = 0
        objective = self.instance.get_objective()
        for i, element in enumerate(self.machine.job_schedule):
            job_id, startTime, endTime = element
            # Test End Time + start Time
            if hasattr(self.instance, 'R'):
                expected_start_time = max(self.instance.R[job_id], ci)
            else:
                expected_start_time = ci
            if hasattr(self.instance, 'S'):
                if prev_job is None:
                    setup_time = self.instance.S[job_id][job_id]
                else:
                    setup_time = self.instance.S[prev_job][job_id]
            else:
                setup_time = 0

            proc_time = self.instance.P[job_id]
            ci = expected_start_time + proc_time + setup_time

            if startTime != expected_start_time or endTime != ci:
                if startTime > expected_start_time and endTime - startTime == proc_time + setup_time :
                    if verbosity : warnings.warn(f'## Warning: found {element} could have been scheduled earlier to reduce idle time')
                else :
                    if verbosity : print(f'## Error:  found {element} expected {job_id,expected_start_time, ci}')
                    is_valid = False

            if objective == Objective.Cmax:
                obj = ci
            elif objective == Objective.wiCi:
                obj += self.instance.W[job_id] * ci
            elif objective == Objective.wiTi:
                obj += self.instance.W[job_id] * max(ci-self.instance.D[job_id], 0)
            elif objective == Objective.wiFi:
                obj += self.instance.W[job_id] * (ci-self.instance.R[job_id])
            elif objective == Objective.Lmax:
                obj = max(obj, ci - self.instance.D[job_id])

            set_jobs.add(job_id)
            prev_job = job_id

        if obj != self.objective_value:
            print(f'## Error: in solution' +
                    f' found objective_value = {self.objective_value} expected {obj}')
            is_valid = False

        if len(set_jobs) != self.instance.n:
            print(f'## Error: in number of jobs' +
                    f' found {len(set_jobs)} job(s) expected {self.instance.n}')
            is_valid = False

        return is_valid


class SM_LocalSearch(Problem.LocalSearch):

    @staticmethod
    def _intra_insertion(solution: SingleSolution):
        """Iterates through the job schedule and try to reschedule every job at a better position to improve the solution

        Args:
            solution (SingleSolution): solution to improve
            objective (RootProblem.Objective): objective to consider

        Returns:
            SingleSolution: improved solution
        """
        for pos in range(len(solution.machine.job_schedule)):
            job = solution.machine.job_schedule[pos]
            old_objective = solution.machine.objective_value
            taken_pos = pos
            for new_pos in range(len(solution.machine.job_schedule)):
                if(pos != new_pos):
                    new_objective = solution.machine.simulate_remove_insert(
                        pos, job.id, new_pos, solution.instance)
                    if new_objective < old_objective:
                        taken_pos = new_pos
                        old_objective = new_objective
            if taken_pos != pos:
                solution.machine.job_schedule.pop(pos)
                solution.machine.job_schedule.insert(taken_pos, job)
                solution.machine.compute_objective(solution.instance, min(taken_pos, pos))
        solution.fix_objective()
        return solution

    @staticmethod
    def _swap(solution: SingleSolution):
        """Iterates through the job schedule and choose the best swap between 2 jobs to improve the solution

        Args:
            solution (SingleSolution): solution to improve
            objective (RootProblem.Objective): objective to consider

        Returns:
            SingleSolution: improved solution
        """
        job_schedule_len = len(solution.machine.job_schedule)
        move = None
        for i in range(0, job_schedule_len):
            for j in range(i+1, job_schedule_len):
                new_ci = solution.machine.simulate_swap(i, j, solution.instance)
                if new_ci < solution.machine.objective_value:
                    if not move:
                        move = (i, j, new_ci)
                    elif new_ci < move[2]:
                        move = (i, j, new_ci)

        if move:
            solution.machine.job_schedule[move[0]], solution.machine.job_schedule[move[1]
                                                                                  ] = solution.machine.job_schedule[move[1]], solution.machine.job_schedule[move[0]]
            solution.machine.objective_value = move[2]
            solution.compute_objective()
        return solution

    def improve(self, solution: SingleSolution) -> SingleSolution:
        """Improves a solution by iteratively calling local search operators

        Args:
            solution (Solution): current solution

        Returns:
            Solution: improved solution
        """
        curr_sol = solution.copy() if self.copy_solution else solution
        for method in self.methods:
            curr_sol = method(curr_sol)

        return curr_sol


class NeighbourhoodGeneration():
    
    @staticmethod
    def select_least_effective(solution: SingleSolution):
        """Select the least effective job according to the objective

        Args:
            solution (SingleSolution): solution to be inspected

        Returns:
            tuple: (lej_pos, lej_id): the position and id of the least effective job
        """
        machine_schedule = solution.machine.job_schedule
        old_objective = solution.machine.objective_value

        # Select the least effective job
        least_effective_pos, least_effective_job, impact = (-1, -1, None) 
        for pos_remove in range(len(machine_schedule)):
            new_objective = solution.machine.simulate_remove_insert(pos_remove, -1, -1, solution.instance)
            if impact is None or old_objective - new_objective < impact:
                least_effective_pos, least_effective_job = (pos_remove, machine_schedule[pos_remove].id)
        
        return least_effective_pos, least_effective_job

    @staticmethod
    def LEJ_insert(solution: SingleSolution, force_improve: bool = True, inplace: bool = True):
        """Applies the best insertion operator on the least effective job on the objective

        Args:
            solution (SingleSolution): solution to be improved
            force_improve (bool, optional): if true, it applies the move only if it improved the solution. Defaults to True.
            inplace (bool, optional): Whether to modify the solution rather than creating a new one. Defaults to True.
        """
        if not inplace:
            solution_copy = solution.copy()
        else:
            solution_copy = solution

        machine_schedule = solution_copy.machine.job_schedule
        old_objective = solution_copy.machine.objective_value

        # Select the least effective job
        least_effective_pos, least_effective_job = NeighbourhoodGeneration.select_least_effective(solution)

        # Search the best insertion move
        best_insertion = None
        for pos_insert in range(len(machine_schedule)):
            new_objective = solution_copy.machine.simulate_remove_insert(least_effective_pos, least_effective_job, pos_insert, solution_copy.instance)
            if best_insertion is None or old_objective - new_objective > best_insertion[1]:
                best_insertion = ( pos_insert, old_objective - new_objective) 
        
        # Apply the best insertion 
        pos_insert, new_objective = best_insertion
        if not force_improve or (new_objective <= old_objective):
            job_i = machine_schedule.pop(least_effective_pos)
            machine_schedule.insert(pos_insert, job_i)

            solution_copy.machine.compute_objective(solution_copy.instance, min(
                least_effective_pos, pos_insert))
            solution_copy.fix_objective()
        
        return solution_copy
    
    @staticmethod
    def LEJ_swap(solution: SingleSolution, force_improve: bool = True, inplace: bool = True):
        """Applies the best insertion operator on the least effective job on the objective

        Args:
            solution (SingleSolution): solution to be improved
            force_improve (bool, optional): if true, it applies the move only if it improved the solution. Defaults to True.
            inplace (bool, optional): Whether to modify the solution rather than creating a new one. Defaults to True.
        """
        if not inplace:
            solution_copy = solution.copy()
        else:
            solution_copy = solution

        machine_schedule = solution_copy.machine.job_schedule
        old_objective = solution_copy.machine.objective_value

        # Select the least effective job
        least_effective_pos, least_effective_job = NeighbourhoodGeneration.select_least_effective(solution)

        # Search the best swap move
        best_swap = None
        for pos_j in range(len(machine_schedule)):
            new_objective = solution_copy.machine.simulate_swap(least_effective_pos, pos_j, solution_copy.instance)
            if best_swap is None or old_objective - new_objective > best_swap[1]:
                best_swap = ( pos_j, old_objective - new_objective) 
        
        # Apply the best insertion 
        pos_j, new_objective = best_swap
        if not force_improve or (new_objective <= old_objective):

            machine_schedule[least_effective_pos], machine_schedule[pos_j] = machine_schedule[pos_j], machine_schedule[least_effective_pos]
            solution_copy.machine.compute_objective(solution_copy.instance, min(least_effective_pos, pos_j))
            solution_copy.fix_objective()
        
        return solution_copy

    @staticmethod
    def random_insert(solution: SingleSolution, force_improve: bool = True, inplace: bool = True):
        """Applies the best insertion operator on the least effective job

        Args:
            solution (SingleSolution): solution to be improved
            force_improve (bool, optional): if true, it applies the move only if it improved the solution. Defaults to True.
            inplace (bool, optional): Whether to modify the solution rather than creating a new one. Defaults to True.
        """
        if not inplace:
            solution_copy = solution.copy()
        else:
            solution_copy = solution

        machine_schedule = solution_copy.machine.job_schedule
        machine_schedule_len = len(machine_schedule)
        old_objective = solution_copy.machine.objective_value

        # Get the job and the position
        random_job_index = random.randrange(machine_schedule_len)
        random_job_id = machine_schedule[random_job_index].id

        random_pos = random_job_index
        while random_pos == random_job_index:
            random_pos = random.randrange(machine_schedule_len)
        
        # Simulate the insertion
        new_objective = solution_copy.machine.simulate_remove_insert(random_job_index, random_job_id, random_pos, solution_copy.instance)

        # Apply the insertion move
        if not force_improve or (new_objective <= old_objective):
            job_i = machine_schedule.pop(random_job_index)
            machine_schedule.insert(random_pos, job_i)

            solution_copy.machine.compute_objective(solution_copy.instance, min(
                random_job_index, random_pos))
            solution_copy.fix_objective()
        
        return solution_copy

    @staticmethod
    def random_swap(solution: SingleSolution, force_improve: bool = True, inplace: bool = True):
        """Performs a random swap between 2 jobs

        Args:
            solution (SingleSolution): Solution to be improved
            objective (RootProblem.Objective) : objective to consider
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.

        Returns:
            SingleSolution: New solution
        """
        if not inplace:
            solution_copy = solution.copy()
        else:
            solution_copy = solution

        # Select the two different random jobs to be swapped 
        machine_schedule = solution_copy.machine.job_schedule
        machine_schedule_len = len(machine_schedule)
        old_objective = solution_copy.machine.objective_value

        random_job_index = random.randrange(machine_schedule_len)
        other_job_index = random.randrange(machine_schedule_len)

        while other_job_index == random_job_index:
            other_job_index = random.randrange(machine_schedule_len)

        # Simulate applying the swap move
        new_objective = solution_copy.machine.simulate_swap(
            random_job_index, other_job_index, solution_copy.instance)

        # Apply the move
        if not force_improve or (new_objective <= old_objective):
            machine_schedule[random_job_index], machine_schedule[
                other_job_index] = machine_schedule[
                    other_job_index], machine_schedule[random_job_index]

            solution_copy.machine.compute_objective(solution_copy.instance, min(
                random_job_index, other_job_index))
            solution_copy.fix_objective()

        return solution_copy

    @staticmethod
    def passive_swap(solution: SingleSolution, force_improve: bool = True):
        """Performs a swap between the 2 least effective jobs in terms of WSPT rule

        Args:
            solution (SingleSolution): Solution to be improved
            force_improve (bool, optional): If true, to apply the move, it must improve the solution. Defaults to True.

        Returns:
            SingleSolution: New solution
        """
        if(len(solution.machine.job_schedule) > 1):

            jobs_list = list(range(solution.instance.n))
            jobs_list.sort(key=lambda job_id: float(
                solution.instance.W[job_id])/float(solution.instance.P[job_id]))
            job_i_id, job_j_id = jobs_list[0], jobs_list[1]
            machine_schedule = solution.machine.job_schedule
            machine_schedule_jobs_id = [job.id for job in machine_schedule]
            job_i_pos, job_j_pos = machine_schedule_jobs_id.index(
                job_i_id), machine_schedule_jobs_id.index(job_j_id)

            old_ci = solution.machine.objective_value

            new_ci = solution.machine.simulate_swap(
                job_i_pos, job_j_pos, solution.instance)

            # Apply the move
            if not force_improve or (new_ci <= old_ci):
                machine_schedule[job_i_pos], machine_schedule[
                    job_j_pos] = machine_schedule[
                        job_j_pos], machine_schedule[job_i_pos]
                solution.machine.compute_objective(
                    solution.instance, min(job_i_pos, job_j_pos))
                solution.fix_objective()

        return solution

    @staticmethod
    def LEJ_neighbour(solution: SingleSolution):
        """Generates a neighbour solution of the given solution for the lahc metaheuristic

        Args:
            solution_i (SingleSolution): Solution to be improved

        Returns:
            SingleSolution: New solution
        """
        r = random.random()
        if r < 0.5:
            solution = NeighbourhoodGeneration.LEJ_insert(
                solution, force_improve=False)
        else:
            solution = NeighbourhoodGeneration.LEJ_swap(
                solution, force_improve=False)

        return solution

    @staticmethod
    def lahc_neighbour(solution: SingleSolution):
        """Generates a neighbour solution of the given solution for the lahc metaheuristic

        Args:
            solution_i (SingleSolution): Solution to be improved

        Returns:
            SingleSolution: New solution
        """
        r = random.random()
        if r < 0.5:
            solution = NeighbourhoodGeneration.random_swap(
                solution, force_improve=False)
        else:
            solution = NeighbourhoodGeneration.random_insert(
                solution, force_improve=False)

        return solution
