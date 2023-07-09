import pyscheduling.JS.JobShop as js
from pyscheduling.core.base_solvers import BaseSolver
from pyscheduling.Problem import Job


class BIBA(BaseSolver):

    def solve(self, instance: js.JobShopInstance):
        """the greedy constructive heuristic (Best insertion based approach) to find an initial solution of a Jobshop instance.

        Args:
            instance (JobShopInstance): Instance to be solved by the heuristic

        Returns:
            Problem.SolveResult: the solver result of the execution of the heuristic
        """
        self.notify_on_start()
        solution = js.JobShopSolution(instance)
        remaining_jobs = set(range(instance.n))
        jobs_timeline = [(0,0) for i in range(instance.n)]
        curr_obj = 0

        while len(remaining_jobs) != 0:
            min_factor = None
            for i in remaining_jobs:
                oper_idx, last_t = jobs_timeline[i]
                m_id, proc_time = instance.P[i][oper_idx]

                start_time, end_time = solution.simulate_insert_last(i, oper_idx, last_t)
                factor = solution.simulate_insert_objective(i, start_time, end_time)
                if not min_factor or (min_factor > factor):
                    min_factor = factor
                    taken_job = i
                    taken_machine = m_id

            # Insert taken job at the end of taken machine
            curr_machine = solution.machines[taken_machine]
            curr_machine.job_schedule.append( Job(taken_job, start_time, end_time) )
            curr_machine.last_job = taken_job

            # Update the job_schedule structure (overall schedule per job)
            _, old_start, old_end = solution.job_schedule.get(taken_job, Job(taken_job, 0, 0))
            solution.job_schedule[taken_job] = Job(taken_job, min(old_start, start_time), max(old_end, end_time))

            # Update job timeline
            oper_idx, last_t = jobs_timeline[taken_job]
            jobs_timeline[taken_job] = (oper_idx + 1, max(end_time, last_t))
            if jobs_timeline[taken_job][0] == len(instance.P[taken_job]):
                remaining_jobs.remove(taken_job)
        
        solution.compute_objective()

        self.notify_on_solution_found(solution)
        self.notify_on_complete()

        return self.solve_result 
    