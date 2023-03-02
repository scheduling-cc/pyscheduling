import os
import sys

import pytest
from pyscheduling.PMSP import RmriSijkCmax, RmSijkCmax, RmridiSijkWiTi, RmriSijkWiCi, RmriSijkWiFi, PM_methods
from pyscheduling.PMSP.ParallelMachines import ParallelSolution, PM_LocalSearch
from pyscheduling.Problem import SolveResult, SolveStatus

# Helper functions
def check_solve_result(solve_result, expected_nb_sol=None):
    is_valid = ParallelSolution.is_valid(solve_result.best_solution)
    assert isinstance(solve_result, SolveResult)
    assert solve_result.solve_status == SolveStatus.FEASIBLE or solve_result.solve_status == SolveStatus.OPTIMAL, f"Solve status should be Feasible instead of {solve_result.solve_status}"
    assert isinstance(solve_result.best_solution,
                      ParallelSolution), f"The returned solution should be of type {ParallelSolution}"
    if expected_nb_sol:
        assert len(
            solve_result.all_solutions) == expected_nb_sol, f"This method should return {expected_nb_sol} solution instead of {len(solve_result.all_solutions)}"
    assert is_valid == True, f"The returned solution is not valid"


class TestRmSijkCmax:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = RmSijkCmax.RmSijkCmax_Instance.generate_random(
            self.n, self.m)
        return instance

    @pytest.fixture
    def instance_zero_file(self, instance_zero):
        instance_zero.to_txt(self.instance_zero_file_path)
        yield self.instance_zero_file_path
        try:
            os.unlink(self.instance_zero_file_path)
        except:
            pass

    # Testing Instance
    def test_generator(self):
        instance = RmSijkCmax.RmSijkCmax_Instance.generate_random(
            self.n, self.m)
        assert isinstance(
            instance, RmSijkCmax.RmSijkCmax_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = RmSijkCmax.RmSijkCmax_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, RmSijkCmax.RmSijkCmax_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = RmSijkCmax.RmSijkCmax_Instance.read_txt(unexisting_path)
        assert e.type == FileNotFoundError

    def test_write_instance_to_file(self, instance_zero):
        file_path = "tests/tmp/instance_zero_test.txt"
        instance_zero.to_txt(file_path)
        assert os.path.exists(file_path)
        try:
            os.unlink(file_path)
        except:
            pass

    # Testing methods
    def test_constructive_solver(self, instance_zero):
        solve_result = RmSijkCmax.Heuristics.BIBA(instance_zero)
        check_solve_result(solve_result, expected_nb_sol=1)

    def test_list_heuristics(self, instance_zero):
        for decreasing in [False, True]:
            for rule_number in range(1, 30):
                solve_result = RmSijkCmax.Heuristics.list_heuristic(
                    instance_zero, rule_number, decreasing)
                check_solve_result(solve_result, expected_nb_sol=1)

    def test_aco_solver(self, instance_zero):
        solve_result = RmSijkCmax.Metaheuristics.antColony(instance_zero)
        check_solve_result(solve_result)

    def test_grasp(self, instance_zero):
        solve_results = PM_methods.Heuristics.grasp(instance_zero)
        check_solve_result(solve_results, expected_nb_sol=5)


    def test_lahc(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.lahc(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_sa(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.SA(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_rsa(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.SA(instance_zero, **{"restricted": True, "time_limit_factor": 0.2})
        check_solve_result(solve_result)

    def test_csp(self, instance_zero):
        if "docplex" in sys.modules:
            solve_result = RmSijkCmax.ExactSolvers.csp(instance_zero, **{"time_limit":10})
            check_solve_result(solve_result)

    # Testing local search
    def test_local_search(self, instance_zero):
        solution = RmSijkCmax.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = PM_LocalSearch(copy_solution=False)
        improved_solution = ls_proc.improve(solution)
        is_valid = ParallelSolution.is_valid(improved_solution)
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, ParallelSolution),  f"The returned solution should be of type {ParallelSolution}"
        assert improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

    def test_local_search_copy(self, instance_zero):
        solution = RmSijkCmax.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = PM_LocalSearch(copy_solution=True)
        improved_solution = ls_proc.improve(solution)
        is_valid = ParallelSolution.is_valid(improved_solution)
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, ParallelSolution),  f"The returned solution should be of type {ParallelSolution}"
        assert not improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

class TestRmriSijkCmax:
    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = RmriSijkCmax.RmriSijkCmax_Instance.generate_random(
            self.n, self.m)
        return instance

    @pytest.fixture
    def instance_zero_file(self, instance_zero):
        instance_zero.to_txt(self.instance_zero_file_path)
        yield self.instance_zero_file_path
        try:
            os.unlink(self.instance_zero_file_path)
        except:
            pass

    # Testing Instance
    def test_generator(self):
        instance = RmriSijkCmax.RmriSijkCmax_Instance.generate_random(
            self.n, self.m)
        assert isinstance(
            instance, RmriSijkCmax.RmriSijkCmax_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = RmriSijkCmax.RmriSijkCmax_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, RmriSijkCmax.RmriSijkCmax_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = RmriSijkCmax.RmriSijkCmax_Instance.read_txt(unexisting_path)
        assert e.type == FileNotFoundError

    def test_write_instance_to_file(self, instance_zero):
        file_path = "tests/tmp/instance_zero_test.txt"
        instance_zero.to_txt(file_path)
        assert os.path.exists(file_path)
        try:
            os.unlink(file_path)
        except:
            pass

    # Testing methods
    def test_constructive_solver(self, instance_zero):
        solve_result = RmriSijkCmax.Heuristics.BIBA(instance_zero)
        check_solve_result(solve_result, expected_nb_sol=1)

    def test_list_heuristics(self, instance_zero):
        for decreasing in [False, True]:
            for rule_number in range(1, 39):
                solve_result = RmriSijkCmax.Heuristics.list_heuristic(
                    instance_zero, rule_number, decreasing)
                check_solve_result(solve_result, expected_nb_sol=1)

    def test_grasp(self, instance_zero):
        solve_results = PM_methods.Heuristics.grasp(instance_zero)
        check_solve_result(solve_results, expected_nb_sol=5)

    def test_lahc(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.lahc(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_sa(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.SA(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_rsa(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.SA(instance_zero, **{"restricted": True, "time_limit_factor": 0.2})
        check_solve_result(solve_result)

    def test_ga(self, instance_zero):
        solve_result = RmriSijkCmax.Metaheuristics.GA(instance_zero, **{"n_iterations": 10})
        print(solve_result)
        check_solve_result(solve_result)

    def test_csp(self, instance_zero):
        if "docplex" in sys.modules:
            solve_result = RmriSijkCmax.ExactSolvers.csp(instance_zero, **{"time_limit":10})
            check_solve_result(solve_result)

    def test_milp(self, instance_zero):
        if "gurobipy" in sys.modules:
            solve_result = RmriSijkCmax.ExactSolvers.milp(instance_zero, **{"time_limit":30})
            check_solve_result(solve_result)

    # Testing local search
    def test_local_search(self, instance_zero):
        solution = RmriSijkCmax.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = PM_LocalSearch(copy_solution=False)
        improved_solution = ls_proc.improve(solution)
        is_valid = ParallelSolution.is_valid(improved_solution)
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, ParallelSolution),  f"The returned solution should be of type {ParallelSolution}"
        assert improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

    def test_local_search_copy(self, instance_zero):
        solution = RmriSijkCmax.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = PM_LocalSearch(copy_solution=True)
        improved_solution = ls_proc.improve(solution)
        is_valid = ParallelSolution.is_valid(improved_solution)
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, ParallelSolution),  f"The returned solution should be of type {ParallelSolution}"
        assert not improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

class TestRmridiSijkWiTi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = RmridiSijkWiTi.RmridiSijkWiTi_Instance.generate_random(
            self.n, self.m)
        return instance

    @pytest.fixture
    def instance_zero_file(self, instance_zero):
        instance_zero.to_txt(self.instance_zero_file_path)
        yield self.instance_zero_file_path
        try:
            os.unlink(self.instance_zero_file_path)
        except:
            pass

    # Testing Instance
    def test_generator(self):
        instance = RmridiSijkWiTi.RmridiSijkWiTi_Instance.generate_random(
            self.n, self.m)
        assert isinstance(
            instance, RmridiSijkWiTi.RmridiSijkWiTi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = RmridiSijkWiTi.RmridiSijkWiTi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, RmridiSijkWiTi.RmridiSijkWiTi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = RmridiSijkWiTi.RmridiSijkWiTi_Instance.read_txt(unexisting_path)
        assert e.type == FileNotFoundError

    def test_write_instance_to_file(self, instance_zero):
        file_path = "tests/tmp/instance_zero_test.txt"
        instance_zero.to_txt(file_path)
        assert os.path.exists(file_path)
        try:
            os.unlink(file_path)
        except:
            pass

    # Testing methods
    def test_constructive_solver(self, instance_zero):
        solve_result = RmridiSijkWiTi.Heuristics.BIBA(instance_zero)
        check_solve_result(solve_result, expected_nb_sol=1)

    def test_list_heuristics(self, instance_zero):
        for decreasing in [False, True]:
            for rule_number in range(1, 13):
                solve_result = RmridiSijkWiTi.Heuristics.list_heuristic(
                    instance_zero, rule_number, decreasing)
                check_solve_result(solve_result, expected_nb_sol=1)

    def test_grasp(self, instance_zero):
        solve_results = PM_methods.Heuristics.grasp(instance_zero)
        check_solve_result(solve_results, expected_nb_sol=5)

    def test_lahc(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.lahc(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_sa(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.SA(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_rsa(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.SA(instance_zero, **{"restricted": True, "time_limit_factor": 0.2})
        check_solve_result(solve_result)

    #def test_csp(self, instance_zero):
    #    if "docplex" in sys.modules:
    #        solve_result = RmSijkCmax.ExactSolvers.csp(instance_zero, **{"time_limit":10})
    #        check_solve_result(solve_result)

    # Testing local search
    def test_local_search(self, instance_zero):
        solution = RmridiSijkWiTi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = PM_LocalSearch(copy_solution=False)
        improved_solution = ls_proc.improve(solution)
        is_valid = ParallelSolution.is_valid(improved_solution)
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, ParallelSolution),  f"The returned solution should be of type {ParallelSolution}"
        assert improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

    def test_local_search_copy(self, instance_zero):
        solution = RmridiSijkWiTi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = PM_LocalSearch(copy_solution=True)
        improved_solution = ls_proc.improve(solution)
        is_valid = ParallelSolution.is_valid(improved_solution)
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, ParallelSolution),  f"The returned solution should be of type {ParallelSolution}"
        assert not improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

class TestRmriSijkWiCi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = RmriSijkWiCi.RmriSijkWiCi_Instance.generate_random(
            self.n, self.m)
        return instance

    @pytest.fixture
    def instance_zero_file(self, instance_zero):
        instance_zero.to_txt(self.instance_zero_file_path)
        yield self.instance_zero_file_path
        try:
            os.unlink(self.instance_zero_file_path)
        except:
            pass

    # Testing Instance
    def test_generator(self):
        instance = RmriSijkWiCi.RmriSijkWiCi_Instance.generate_random(
            self.n, self.m)
        assert isinstance(
            instance, RmriSijkWiCi.RmriSijkWiCi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = RmriSijkWiCi.RmriSijkWiCi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, RmriSijkWiCi.RmriSijkWiCi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = RmriSijkWiCi.RmriSijkWiCi_Instance.read_txt(unexisting_path)
        assert e.type == FileNotFoundError

    def test_write_instance_to_file(self, instance_zero):
        file_path = "tests/tmp/instance_zero_test.txt"
        instance_zero.to_txt(file_path)
        assert os.path.exists(file_path)
        try:
            os.unlink(file_path)
        except:
            pass

    # Testing methods
    def test_constructive_solver(self, instance_zero):
        solve_result = RmriSijkWiCi.Heuristics.BIBA(instance_zero)
        check_solve_result(solve_result, expected_nb_sol=1)

    def test_list_heuristics(self, instance_zero):
        for decreasing in [False, True]:
            for rule_number in range(1, 23):
                solve_result = RmriSijkWiCi.Heuristics.list_heuristic(
                    instance_zero, rule_number, decreasing)
                check_solve_result(solve_result, expected_nb_sol=1)

    def test_grasp(self, instance_zero):
        solve_results = PM_methods.Heuristics.grasp(instance_zero)
        check_solve_result(solve_results, expected_nb_sol=5)

    def test_lahc(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.lahc(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_sa(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.SA(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_rsa(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.SA(instance_zero, **{"restricted": True, "time_limit_factor": 0.2})
        check_solve_result(solve_result)

    #def test_csp(self, instance_zero):
    #    if "docplex" in sys.modules:
    #        solve_result = RmSijkCmax.ExactSolvers.csp(instance_zero, **{"time_limit":10})
    #        check_solve_result(solve_result)

    # Testing local search
    def test_local_search(self, instance_zero):
        solution = RmriSijkWiCi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = PM_LocalSearch(copy_solution=False)
        improved_solution = ls_proc.improve(solution)
        is_valid = ParallelSolution.is_valid(improved_solution)
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, ParallelSolution),  f"The returned solution should be of type {ParallelSolution}"
        assert improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

    def test_local_search_copy(self, instance_zero):
        solution = RmriSijkWiCi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = PM_LocalSearch(copy_solution=True)
        improved_solution = ls_proc.improve(solution)
        is_valid = ParallelSolution.is_valid(improved_solution)
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, ParallelSolution),  f"The returned solution should be of type {ParallelSolution}"
        assert not improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'
        
class TestRmriSijkWiFi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = RmriSijkWiFi.RmriSijkWiFi_Instance.generate_random(
            self.n, self.m)
        return instance

    @pytest.fixture
    def instance_zero_file(self, instance_zero):
        instance_zero.to_txt(self.instance_zero_file_path)
        yield self.instance_zero_file_path
        try:
            os.unlink(self.instance_zero_file_path)
        except:
            pass

    # Testing Instance
    def test_generator(self):
        instance = RmriSijkWiFi.RmriSijkWiFi_Instance.generate_random(
            self.n, self.m)
        assert isinstance(
            instance, RmriSijkWiFi.RmriSijkWiFi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = RmriSijkWiFi.RmriSijkWiFi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, RmriSijkWiFi.RmriSijkWiFi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = RmriSijkWiFi.RmriSijkWiFi_Instance.read_txt(unexisting_path)
        assert e.type == FileNotFoundError

    def test_write_instance_to_file(self, instance_zero):
        file_path = "tests/tmp/instance_zero_test.txt"
        instance_zero.to_txt(file_path)
        assert os.path.exists(file_path)
        try:
            os.unlink(file_path)
        except:
            pass

    # Testing methods
    def test_constructive_solver(self, instance_zero):
        solve_result = RmriSijkWiFi.Heuristics.BIBA(instance_zero)
        check_solve_result(solve_result, expected_nb_sol=1)

    def test_list_heuristics(self, instance_zero):
        for decreasing in [False, True]:
            for rule_number in range(1, 23):
                solve_result = RmriSijkWiFi.Heuristics.list_heuristic(
                    instance_zero, rule_number, decreasing)
                check_solve_result(solve_result, expected_nb_sol=1)

    def test_grasp(self, instance_zero):
        solve_results = PM_methods.Heuristics.grasp(instance_zero)
        check_solve_result(solve_results, expected_nb_sol=5)

    def test_lahc(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.lahc(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_sa(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.SA(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_rsa(self, instance_zero):
        solve_result = PM_methods.Metaheuristics.SA(instance_zero, **{"restricted": True, "time_limit_factor": 0.2})
        check_solve_result(solve_result)

    #def test_csp(self, instance_zero):
    #    if "docplex" in sys.modules:
    #        solve_result = RmSijkCmax.ExactSolvers.csp(instance_zero, **{"time_limit":10})
    #        check_solve_result(solve_result)

    # Testing local search
    def test_local_search(self, instance_zero):
        solution = RmriSijkWiFi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = PM_LocalSearch(copy_solution=False)
        improved_solution = ls_proc.improve(solution)
        is_valid = ParallelSolution.is_valid(improved_solution)
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, ParallelSolution),  f"The returned solution should be of type {ParallelSolution}"
        assert improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

    def test_local_search_copy(self, instance_zero):
        solution = RmriSijkWiFi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = PM_LocalSearch(copy_solution=True)
        improved_solution = ls_proc.improve(solution)
        is_valid = ParallelSolution.is_valid(improved_solution)
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, ParallelSolution),  f"The returned solution should be of type {ParallelSolution}"
        assert not improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'