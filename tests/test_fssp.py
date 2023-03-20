import os
import sys

import pytest

from pyscheduling.FS import (FmridiSijkwiTi, FmriSijkCmax, FmriSijkwiCi,
                             FmriSijkwiFi, FS_methods)
from pyscheduling.FS.FlowShop import FlowShopSolution, FS_LocalSearch
from pyscheduling.Problem import Objective, SolveResult, SolveStatus


# Helper functions
def check_solve_result(solve_result, expected_nb_sol=None):
    is_valid = solve_result.best_solution.is_valid()
    assert isinstance(solve_result, SolveResult)
    assert solve_result.solve_status == SolveStatus.FEASIBLE or solve_result.solve_status == SolveStatus.OPTIMAL, f"Solve status should be Feasible instead of {solve_result.solve_status}"
    assert isinstance(solve_result.best_solution, FlowShopSolution), f"The returned solution should be of type {FlowShopSolution}"
    if expected_nb_sol:
        assert len(
            solve_result.all_solutions) == expected_nb_sol, f"This method should return {expected_nb_sol} solution instead of {len(solve_result.all_solutions)}"
    assert is_valid == True, f"The returned solution is not valid"


def instance_check(instance, instance_class, attr_list, objective):
    assert isinstance(instance, instance_class), f'Instance is not of the right type'
    for attr in attr_list:
        assert hasattr(instance, attr), f'Instance does not have attribute {attr}'
    assert instance.get_objective() == objective, f'Expected objective {objective}, got {instance.get_objective()} instead'

default_rule = lambda instance, job_id: sum(instance.P[job_id])

class TestFmriSijkCmax:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = FmriSijkCmax.FmriSijkCmax_Instance.generate_random(self.n, self.m)
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
        instance = FmriSijkCmax.FmriSijkCmax_Instance.generate_random(self.n, self.m)
        instance_check(instance, FmriSijkCmax.FmriSijkCmax_Instance, ["P", "R", "S"], Objective.Cmax)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'
        

    def test_read_existing_file(self, instance_zero_file):
        instance = FmriSijkCmax.FmriSijkCmax_Instance.read_txt(instance_zero_file)
        instance_check(instance, FmriSijkCmax.FmriSijkCmax_Instance, ["P", "R", "S"], Objective.Cmax)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = FmriSijkCmax.FmriSijkCmax_Instance.read_txt(unexisting_path)
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
    def test_biba(self, instance_zero):
        solve_result = FmriSijkCmax.Heuristics.BIBA(instance_zero)
        check_solve_result(solve_result, expected_nb_sol=1)

    def test_list_heuristics(self, instance_zero):
        for reverse in [False, True]:
            solve_result = FmriSijkCmax.Heuristics.dispatch_heuristic(
                instance_zero, default_rule, reverse)
            check_solve_result(solve_result, expected_nb_sol=1)

    def test_grasp(self, instance_zero):
        solve_results = FmriSijkCmax.Heuristics.grasp(instance_zero, n_iterations = 5)
        check_solve_result(solve_results, expected_nb_sol=5)

    def test_lahc(self, instance_zero):
        solve_result = FmriSijkCmax.Metaheuristics.lahc(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_sa(self, instance_zero):
        solve_result = FmriSijkCmax.Metaheuristics.SA(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)

    # Testing local search
    def test_local_search(self, instance_zero):
        solution = FmriSijkCmax.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = FS_LocalSearch(copy_solution=False)
        improved_solution = ls_proc.improve(solution)

        is_valid = improved_solution.is_valid()
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, FlowShopSolution),  f"The returned solution should be of type {FlowShopSolution}"
        assert improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

    def test_local_search_copy(self, instance_zero):
        solution = FmriSijkCmax.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = FS_LocalSearch(copy_solution=True)
        improved_solution = ls_proc.improve(solution)

        is_valid = improved_solution.is_valid()
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, FlowShopSolution),  f"The returned solution should be of type {FlowShopSolution}"
        assert not improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

class TestFmriSijkwiCi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = FmriSijkwiCi.FmriSijkwiCi_Instance.generate_random(self.n, self.m)
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
        instance = FmriSijkwiCi.FmriSijkwiCi_Instance.generate_random(self.n, self.m)
        instance_check(instance, FmriSijkwiCi.FmriSijkwiCi_Instance, ["P", "R", "S", "W"], Objective.wiCi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'
        

    def test_read_existing_file(self, instance_zero_file):
        instance = FmriSijkwiCi.FmriSijkwiCi_Instance.read_txt(instance_zero_file)
        instance_check(instance, FmriSijkwiCi.FmriSijkwiCi_Instance, ["P", "R", "S", "W"], Objective.wiCi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = FmriSijkwiCi.FmriSijkwiCi_Instance.read_txt(unexisting_path)
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
    def test_biba(self, instance_zero):
        solve_result = FmriSijkwiCi.Heuristics.BIBA(instance_zero)
        check_solve_result(solve_result, expected_nb_sol=1)

    def test_list_heuristics(self, instance_zero):
        for reverse in [False, True]:
            solve_result = FmriSijkwiCi.Heuristics.dispatch_heuristic(
                instance_zero, default_rule, reverse)
            check_solve_result(solve_result, expected_nb_sol=1)

    def test_grasp(self, instance_zero):
        solve_results = FmriSijkwiCi.Heuristics.grasp(instance_zero, n_iterations = 5)
        check_solve_result(solve_results, expected_nb_sol=5)

    def test_lahc(self, instance_zero):
        solve_result = FmriSijkwiCi.Metaheuristics.lahc(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_sa(self, instance_zero):
        solve_result = FmriSijkwiCi.Metaheuristics.SA(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)

    # Testing local search
    def test_local_search(self, instance_zero):
        solution = FmriSijkwiCi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = FS_LocalSearch(copy_solution=False)
        improved_solution = ls_proc.improve(solution)

        is_valid = improved_solution.is_valid()
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, FlowShopSolution),  f"The returned solution should be of type {FlowShopSolution}"
        assert improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

    def test_local_search_copy(self, instance_zero):
        solution = FmriSijkwiCi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = FS_LocalSearch(copy_solution=True)
        improved_solution = ls_proc.improve(solution)

        is_valid = improved_solution.is_valid()
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, FlowShopSolution),  f"The returned solution should be of type {FlowShopSolution}"
        assert not improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

class TestFmriSijkwiFi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = FmriSijkwiFi.FmriSijkwiFi_Instance.generate_random(self.n, self.m)
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
        instance = FmriSijkwiFi.FmriSijkwiFi_Instance.generate_random(self.n, self.m)
        instance_check(instance, FmriSijkwiFi.FmriSijkwiFi_Instance, ["P", "R", "S", "W"], Objective.wiFi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'
        

    def test_read_existing_file(self, instance_zero_file):
        instance = FmriSijkwiFi.FmriSijkwiFi_Instance.read_txt(instance_zero_file)
        instance_check(instance, FmriSijkwiFi.FmriSijkwiFi_Instance, ["P", "R", "S", "W"], Objective.wiFi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = FmriSijkwiFi.FmriSijkwiFi_Instance.read_txt(unexisting_path)
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
    def test_biba(self, instance_zero):
        solve_result = FmriSijkwiFi.Heuristics.BIBA(instance_zero)
        check_solve_result(solve_result, expected_nb_sol=1)

    def test_list_heuristics(self, instance_zero):
        for reverse in [False, True]:
            solve_result = FmriSijkwiFi.Heuristics.dispatch_heuristic(
                instance_zero, default_rule, reverse)
            check_solve_result(solve_result, expected_nb_sol=1)

    def test_grasp(self, instance_zero):
        solve_results = FmriSijkwiFi.Heuristics.grasp(instance_zero, n_iterations = 5)
        check_solve_result(solve_results, expected_nb_sol=5)

    def test_lahc(self, instance_zero):
        solve_result = FmriSijkwiFi.Metaheuristics.lahc(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_sa(self, instance_zero):
        solve_result = FmriSijkwiFi.Metaheuristics.SA(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)

    # Testing local search
    def test_local_search(self, instance_zero):
        solution = FmriSijkwiFi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = FS_LocalSearch(copy_solution=False)
        improved_solution = ls_proc.improve(solution)

        is_valid = improved_solution.is_valid()
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, FlowShopSolution),  f"The returned solution should be of type {FlowShopSolution}"
        assert improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

    def test_local_search_copy(self, instance_zero):
        solution = FmriSijkwiFi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = FS_LocalSearch(copy_solution=True)
        improved_solution = ls_proc.improve(solution)

        is_valid = improved_solution.is_valid()
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, FlowShopSolution),  f"The returned solution should be of type {FlowShopSolution}"
        assert not improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

class TestFmridiSijkwiTi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = FmridiSijkwiTi.FmridiSijkwiTi_Instance.generate_random(self.n, self.m)
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
        instance = FmridiSijkwiTi.FmridiSijkwiTi_Instance.generate_random(self.n, self.m)
        instance_check(instance, FmridiSijkwiTi.FmridiSijkwiTi_Instance, ["P", "R", "S", "W", "D"], Objective.wiTi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'
        

    def test_read_existing_file(self, instance_zero_file):
        instance = FmridiSijkwiTi.FmridiSijkwiTi_Instance.read_txt(instance_zero_file)
        instance_check(instance, FmridiSijkwiTi.FmridiSijkwiTi_Instance, ["P", "R", "S", "W", "D"], Objective.wiTi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = FmridiSijkwiTi.FmridiSijkwiTi_Instance.read_txt(unexisting_path)
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
    def test_biba(self, instance_zero):
        solve_result = FmridiSijkwiTi.Heuristics.BIBA(instance_zero)
        check_solve_result(solve_result, expected_nb_sol=1)

    def test_list_heuristics(self, instance_zero):
        for reverse in [False, True]:
            solve_result = FmridiSijkwiTi.Heuristics.dispatch_heuristic(
                instance_zero, default_rule, reverse)
            check_solve_result(solve_result, expected_nb_sol=1)

    def test_grasp(self, instance_zero):
        solve_results = FmridiSijkwiTi.Heuristics.grasp(instance_zero, n_iterations = 5)
        check_solve_result(solve_results, expected_nb_sol=5)

    def test_lahc(self, instance_zero):
        solve_result = FmridiSijkwiTi.Metaheuristics.lahc(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)
    
    def test_sa(self, instance_zero):
        solve_result = FmridiSijkwiTi.Metaheuristics.SA(instance_zero, **{"time_limit_factor": 0.2})
        check_solve_result(solve_result)

    # Testing local search
    def test_local_search(self, instance_zero):
        solution = FmridiSijkwiTi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = FS_LocalSearch(copy_solution=False)
        improved_solution = ls_proc.improve(solution)

        is_valid = improved_solution.is_valid()
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, FlowShopSolution),  f"The returned solution should be of type {FlowShopSolution}"
        assert improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'

    def test_local_search_copy(self, instance_zero):
        solution = FmridiSijkwiTi.Heuristics.BIBA(instance_zero).best_solution
        ls_proc = FS_LocalSearch(copy_solution=True)
        improved_solution = ls_proc.improve(solution)

        is_valid = improved_solution.is_valid()
        assert is_valid == True, f"The returned solution is not valid"
        assert isinstance(improved_solution, FlowShopSolution),  f"The returned solution should be of type {FlowShopSolution}"
        assert not improved_solution is solution, f'Copy solution is set to False and is not keeping the original solution'