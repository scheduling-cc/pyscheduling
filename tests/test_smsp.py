import os
import sys

import pytest
from pyscheduling.SMSP import *
from pyscheduling.Problem import SolveResult, SolveStatus
from pyscheduling.SMSP.solvers import BIBA, GRASP, SA, LAHC
import pyscheduling.SMSP.SingleMachine as sm

# Helper functions
def check_solve_result(solve_result, expected_nb_sol=None):
    is_valid = SingleMachine.SingleSolution.is_valid(solve_result.best_solution)
    assert isinstance(solve_result, SolveResult)
    assert solve_result.solve_status == SolveStatus.FEASIBLE or solve_result.solve_status == SolveStatus.OPTIMAL, f"Solve status should be Feasible instead of {solve_result.solve_status}"
    assert isinstance(solve_result.best_solution,
                      SingleMachine.SingleSolution), f"The returned solution should be of type {SingleMachine.SingleSolution}"
    if expected_nb_sol:
        assert len(
            solve_result.all_solutions) == expected_nb_sol, f"This method should return {expected_nb_sol} solution instead of {len(solve_result.all_solutions)}"
    assert is_valid == True, f"The returned solution is not valid"

def check_solver(solver, instance, expected_nb_sol=None):
    solve_result = solver.solve(instance)
    check_solve_result(solve_result, expected_nb_sol)

class TestrisijwiCi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n = 20

    @pytest.fixture
    def instance_zero(self):
        instance = risijwiCi.risijwiCi_Instance.generate_random(self.n)
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
        instance = risijwiCi.risijwiCi_Instance.generate_random(self.n)
        assert isinstance(
            instance, risijwiCi.risijwiCi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = risijwiCi.risijwiCi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, risijwiCi.risijwiCi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = risijwiCi.risijwiCi_Instance.read_txt(unexisting_path)
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
        for rule_number in range(0, 4):
            check_solver(risijwiCi.ListHeuristic(rule_number=rule_number),
                        instance_zero, expected_nb_sol=1)

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, expected_nb_sol=1)

    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, expected_nb_sol=5)

    def test_lahc(self, instance_zero):
        check_solver(LAHC(time_limit_factor=0.2), instance_zero)

    def test_sa(self, instance_zero):
        check_solver(SA(time_limit_factor=0.2), instance_zero)

class TestrisijwiFi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n = 20

    @pytest.fixture
    def instance_zero(self):
        instance = risijwiFi.risijwiFi_Instance.generate_random(self.n)
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
        instance = risijwiFi.risijwiFi_Instance.generate_random(
            self.n)
        assert isinstance(
            instance, risijwiFi.risijwiFi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = risijwiFi.risijwiFi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, risijwiFi.risijwiFi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = risijwiFi.risijwiFi_Instance.read_txt(unexisting_path)
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
        for rule_number in range(0, 4):
            check_solver(risijwiFi.ListHeuristic(rule_number=rule_number),
                        instance_zero, expected_nb_sol=1)
            
    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)

    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)

    def test_lahc(self, instance_zero):
        check_solver(LAHC(time_limit_factor=0.2), instance_zero)

    def test_sa(self, instance_zero):
        check_solver(SA(time_limit_factor=0.2), instance_zero)

class TestwiCi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n = 20

    @pytest.fixture
    def instance_zero(self):
        instance = wiCi.wiCi_Instance.generate_random(
            self.n)
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
        instance = wiCi.wiCi_Instance.generate_random(
            self.n)
        assert isinstance(
            instance, wiCi.wiCi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = wiCi.wiCi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, wiCi.wiCi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = wiCi.wiCi_Instance.read_txt(unexisting_path)
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
    def test_WSPT_solver(self, instance_zero):
        check_solver(wiCi.WSPT(), instance_zero, 1)

class TestriwiCi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n = 20

    @pytest.fixture
    def instance_zero(self):
        instance = riwiCi.riwiCi_Instance.generate_random(
            self.n)
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
        instance = riwiCi.riwiCi_Instance.generate_random(
            self.n)
        assert isinstance(
            instance, riwiCi.riwiCi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = riwiCi.riwiCi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, riwiCi.riwiCi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = riwiCi.riwiCi_Instance.read_txt(unexisting_path)
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
    def test_WSECi_solver(self, instance_zero):
        check_solver(riwiCi.WSECi(), instance_zero, 1)
    
    def test_WSAPT_solver(self, instance_zero):
        check_solver(riwiCi.WSAPT(), instance_zero, 1)

    def test_list_heuristics(self, instance_zero):
        for rule_number in range(1, 3):
            check_solver(riwiCi.ListHeuristic(rule_number=rule_number),
                        instance_zero, 1)

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)

    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)

    def test_lahc(self, instance_zero):
        check_solver(LAHC(time_limit_factor=0.2), instance_zero)

    def test_sa(self, instance_zero):
        check_solver(SA(time_limit_factor=0.2), instance_zero)

class TestwiTi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n = 20

    @pytest.fixture
    def instance_zero(self):
        instance = wiTi.wiTi_Instance.generate_random(
            self.n)
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
        instance = wiTi.wiTi_Instance.generate_random(
            self.n)
        assert isinstance(
            instance, wiTi.wiTi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = wiTi.wiTi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, wiTi.wiTi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = wiTi.wiTi_Instance.read_txt(unexisting_path)
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
    def test_WSPT_solver(self, instance_zero):
        check_solver(wiTi.WSPT(), instance_zero, 1)

    def test_MS_solver(self, instance_zero):
        check_solver(wiTi.MS(),instance_zero, 1)

    def test_ACT_solver(self, instance_zero):
        check_solver(wiTi.ACT(),instance_zero, 1)

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)

    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)

    def test_lahc(self, instance_zero):
        check_solver(LAHC(time_limit_factor=0.2), instance_zero)

    def test_sa(self, instance_zero):
        check_solver(SA(time_limit_factor=0.2), instance_zero)

class TestriwiTi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n = 20

    @pytest.fixture
    def instance_zero(self):
        instance = riwiTi.riwiTi_Instance.generate_random(
            self.n)
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
        instance = riwiTi.riwiTi_Instance.generate_random(
            self.n)
        assert isinstance(
            instance, riwiTi.riwiTi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = riwiTi.riwiTi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, riwiTi.riwiTi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = riwiTi.riwiTi_Instance.read_txt(unexisting_path)
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
    def test_ACT_WSECi_solver(self, instance_zero):
        check_solver(riwiTi.ACT_WSECi(), instance_zero, 1)

    def test_ACT_WSAPT_solver(self, instance_zero):
        check_solver(riwiTi.ACT_WSAPT(), instance_zero, 1)

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)
 
    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)

    def test_lahc(self, instance_zero):
        check_solver(LAHC(time_limit_factor=0.2), instance_zero)

    def test_sa(self, instance_zero):
        check_solver(SA(time_limit_factor=0.2), instance_zero)
 
class TestsijwiTi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n = 20

    @pytest.fixture
    def instance_zero(self):
        instance = sijwiTi.sijwiTi_Instance.generate_random(
            self.n)
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
        instance = sijwiTi.sijwiTi_Instance.generate_random(
            self.n)
        assert isinstance(
            instance, sijwiTi.sijwiTi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = sijwiTi.sijwiTi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, sijwiTi.sijwiTi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = sijwiTi.sijwiTi_Instance.read_txt(unexisting_path)
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
    def test_ACTS_solver(self, instance_zero):
        check_solver(sijwiTi.ACTS(),instance_zero, 1)

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)
 
    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)

    def test_lahc(self, instance_zero):
        check_solver(LAHC(time_limit_factor=0.2), instance_zero)

    def test_sa(self, instance_zero):
        check_solver(SA(time_limit_factor=0.2), instance_zero)
 

class TestrisijwiTi:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n = 20

    @pytest.fixture
    def instance_zero(self):
        instance = risijwiTi.risijwiTi_Instance.generate_random(
            self.n)
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
        instance = risijwiTi.risijwiTi_Instance.generate_random(
            self.n)
        assert isinstance(
            instance, risijwiTi.risijwiTi_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = risijwiTi.risijwiTi_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, risijwiTi.risijwiTi_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = risijwiTi.risijwiTi_Instance.read_txt(unexisting_path)
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
    def test_ACTS_WSECi_solver(self, instance_zero):
        check_solver(risijwiTi.ACTS_WSECi(),instance_zero, 1)

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)
 
    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)

    def test_lahc(self, instance_zero):
        check_solver(LAHC(time_limit_factor=0.2), instance_zero)

    def test_sa(self, instance_zero):
        check_solver(SA(time_limit_factor=0.2), instance_zero)
 
class TestsijCmax:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n = 20

    @pytest.fixture
    def instance_zero(self):
        instance = sijCmax.sijCmax_Instance.generate_random(
            self.n)
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
        instance = sijCmax.sijCmax_Instance.generate_random(
            self.n)
        assert isinstance(
            instance, sijCmax.sijCmax_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = sijCmax.sijCmax_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, sijCmax.sijCmax_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = sijCmax.sijCmax_Instance.read_txt(unexisting_path)
        assert e.type == FileNotFoundError

    def test_write_instance_to_file(self, instance_zero):
        file_path = "tests/tmp/instance_zero_test.txt"
        instance_zero.to_txt(file_path)
        assert os.path.exists(file_path)
        try:
            os.unlink(file_path)
        except:
            pass

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)
 
    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)

    def test_lahc(self, instance_zero):
        check_solver(LAHC(time_limit_factor=0.2), instance_zero)

    def test_sa(self, instance_zero):
        check_solver(SA(time_limit_factor=0.2), instance_zero)
 
class TestrisijCmax:

    instance_zero_file_path = "tests/tmp/instance_zero.txt"
    n = 20

    @pytest.fixture
    def instance_zero(self):
        instance = risijCmax.risijCmax_Instance.generate_random(
            self.n)
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
        instance = risijCmax.risijCmax_Instance.generate_random(
            self.n)
        assert isinstance(
            instance, risijCmax.risijCmax_Instance), f'Instance is not of the right type'
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_existing_file(self, instance_zero_file):
        instance = risijCmax.risijCmax_Instance.read_txt(instance_zero_file)
        assert isinstance(instance, risijCmax.risijCmax_Instance)
        assert instance.n == self.n, f'Jobs number is not correct, expected {self.n} got {instance.n} instead'

    def test_read_unexisting_file(self):
        unexisting_path = "tests/tmp/unexisting_file.txt"
        with pytest.raises(Exception) as e:
            instance = risijCmax.risijCmax_Instance.read_txt(unexisting_path)
        assert e.type == FileNotFoundError

    def test_write_instance_to_file(self, instance_zero):
        file_path = "tests/tmp/instance_zero_test.txt"
        instance_zero.to_txt(file_path)
        assert os.path.exists(file_path)
        try:
            os.unlink(file_path)
        except:
            pass

    def test_biba(self, instance_zero):
        check_solver(BIBA(), instance_zero, 1)
 
    def test_grasp(self, instance_zero):
        check_solver(GRASP(), instance_zero, 5)

    def test_lahc(self, instance_zero):
        check_solver(LAHC(time_limit_factor=0.2), instance_zero)

    def test_sa(self, instance_zero):
        check_solver(SA(time_limit_factor=0.2), instance_zero)
 