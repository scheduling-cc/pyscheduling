import os
import sys

import pytest

from pyscheduling.JS import JmridiSijkwiTi, JmriSijkCmax, JmriSijkwiCi,JmriSijkwiFi
from pyscheduling.JS.JobShop import JobShopSolution
from pyscheduling.Problem import Objective, SolveResult, SolveStatus
from pyscheduling.JS.solvers import BIBA, DispatchHeuristic, GRASP


# Helper functions
def check_solve_result(solve_result, expected_nb_sol=None):
    is_valid = solve_result.best_solution.is_valid()
    assert isinstance(solve_result, SolveResult)
    assert solve_result.solve_status == SolveStatus.FEASIBLE or solve_result.solve_status == SolveStatus.OPTIMAL, f"Solve status should be Feasible/Optimal instead of {solve_result.solve_status}"
    assert isinstance(solve_result.best_solution, JobShopSolution), f"The returned solution should be of type {JobShopSolution}"
    if expected_nb_sol:
        assert len(
            solve_result.all_solutions) == expected_nb_sol, f"This method should return {expected_nb_sol} solution instead of {len(solve_result.all_solutions)}"
    assert is_valid == True, f"The returned solution is not valid"

def check_solver(solver, instance, expected_nb_sol=None):
    solve_result = solver.solve(instance)
    check_solve_result(solve_result, expected_nb_sol)

def instance_check(instance, instance_class, attr_list, objective):
    assert isinstance(instance, instance_class), f'Instance is not of the right type'
    for attr in attr_list:
        assert hasattr(instance, attr), f'Instance does not have attribute {attr}'
    assert instance.get_objective() == objective, f'Expected objective {objective}, got {instance.get_objective()} instead'

default_rule = lambda instance, job_tuple: instance.P[job_tuple[0]][job_tuple[1][0]]

class TestJmriSijkCmax:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = JmriSijkCmax.JmriSijkCmax_Instance.generate_random(self.n, self.m)
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
        instance = JmriSijkCmax.JmriSijkCmax_Instance.generate_random(self.n, self.m)
        instance_check(instance, JmriSijkCmax.JmriSijkCmax_Instance, ["P", "R", "S"], Objective.Cmax)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'
        

    def test_read_existing_file(self, instance_zero_file):
        instance = JmriSijkCmax.JmriSijkCmax_Instance.read_txt(instance_zero_file)
        instance_check(instance, JmriSijkCmax.JmriSijkCmax_Instance, ["P", "R", "S"], Objective.Cmax)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = JmriSijkCmax.JmriSijkCmax_Instance.read_txt(unexisting_path)
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
    def test_list_heuristics(self, instance_zero):
        for reverse in [False, True]:
            check_solver(DispatchHeuristic( default_rule, reverse = reverse),
                         instance_zero, 1)

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)

    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)

class TestJmriSijkwiCi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = JmriSijkwiCi.JmriSijkwiCi_Instance.generate_random(self.n, self.m)
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
        instance = JmriSijkwiCi.JmriSijkwiCi_Instance.generate_random(self.n, self.m)
        instance_check(instance, JmriSijkwiCi.JmriSijkwiCi_Instance, ["P", "R", "S", "W"], Objective.wiCi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'
        

    def test_read_existing_file(self, instance_zero_file):
        instance = JmriSijkwiCi.JmriSijkwiCi_Instance.read_txt(instance_zero_file)
        instance_check(instance, JmriSijkwiCi.JmriSijkwiCi_Instance, ["P", "R", "S", "W"], Objective.wiCi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = JmriSijkwiCi.JmriSijkwiCi_Instance.read_txt(unexisting_path)
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
    def test_list_heuristics(self, instance_zero):
        for reverse in [False, True]:
            check_solver(DispatchHeuristic( default_rule, reverse = reverse),
                         instance_zero, 1)

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)

    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)

class TestJmriSijkwiFi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = JmriSijkwiFi.JmriSijkwiFi_Instance.generate_random(self.n, self.m)
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
        instance = JmriSijkwiFi.JmriSijkwiFi_Instance.generate_random(self.n, self.m)
        instance_check(instance, JmriSijkwiFi.JmriSijkwiFi_Instance, ["P", "R", "S", "W"], Objective.wiFi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'
        
    def test_read_existing_file(self, instance_zero_file):
        instance = JmriSijkwiFi.JmriSijkwiFi_Instance.read_txt(instance_zero_file)
        instance_check(instance, JmriSijkwiFi.JmriSijkwiFi_Instance, ["P", "R", "S", "W"], Objective.wiFi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = JmriSijkwiFi.JmriSijkwiFi_Instance.read_txt(unexisting_path)
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
    def test_list_heuristics(self, instance_zero):
        for reverse in [False, True]:
            check_solver(DispatchHeuristic( default_rule, reverse = reverse),
                         instance_zero, 1)

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)

    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)


class TestJmridiSijkwiTi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n, m = 20, 2

    @pytest.fixture
    def instance_zero(self):
        instance = JmridiSijkwiTi.JmridiSijkwiTi_Instance.generate_random(self.n, self.m)
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
        instance = JmridiSijkwiTi.JmridiSijkwiTi_Instance.generate_random(self.n, self.m)
        instance_check(instance, JmridiSijkwiTi.JmridiSijkwiTi_Instance, ["P", "R", "S", "W", "D"], Objective.wiTi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'
        

    def test_read_existing_file(self, instance_zero_file):
        instance = JmridiSijkwiTi.JmridiSijkwiTi_Instance.read_txt(instance_zero_file)
        instance_check(instance, JmridiSijkwiTi.JmridiSijkwiTi_Instance, ["P", "R", "S", "W", "D"], Objective.wiTi)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'
        assert instance.m == self.m, f'Jobs number is not correct, expected {self.m} got {instance.m} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = JmridiSijkwiTi.JmridiSijkwiTi_Instance.read_txt(unexisting_path)
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
    def test_list_heuristics(self, instance_zero):
        for reverse in [False, True]:
            check_solver(DispatchHeuristic(default_rule, reverse = reverse),
                         instance_zero, 1)

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)

    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)
