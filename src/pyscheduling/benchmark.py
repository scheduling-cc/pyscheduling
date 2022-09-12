from enum import Enum
import os
import sys
from pathlib import Path
import time
import csv

import pyscheduling.Problem as Problem

class Log(Enum):
    objective = "objective_value"
    runtime = "runtime"
    nb_solution = "nb_solution"
    status = "solve_status"

def write_excel(fname : Path, result):
    """ Wrapper method to bypass pandas dependancy, mainly used in opale server
    Args:
        fname (str): path to the result excel file, without the .extension
        result_dict (dict): (key, value) pairs of the results 
    """
    keys = result[0].keys()

    with open(fname + ".csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(result)

def run_solver_instance(fname : Path, instances : list[Problem.Instance],methods_args : list[object],kwargs={}, log_param : list[Log] = [Log.objective,Log.runtime]):
    """Performs the benchmark of a list of methods on a list of instances, creates a csv file of the results and
    outputs the results

    Args:
        fname (Path): output filename
        instances (list[Problem.Instance]): List of instances to solve
        methods_args (list[object]): List of either methods or couple of method and its extra-arguments.
        kwargs (dict, optional): Extra-arguments for all methods. Defaults to {}.
        log_param (list[Log], optional): parameters to be logged. Defaults to [Log.objective,Log.runtime].

    Returns:
        dict: result of the benchmark
    """
    #instances_names = [instance.name for instance in instances]
    #methods_names = [method.__name__ for method in methods]
    run_methods_on_instances = []
    for instance in instances :
        run_methods_on_instance = {}
        for method_args in methods_args :
            if type(method_args) == tuple : 
                method = method_args[0]
                args = method_args[1] | kwargs
            else: 
                method = method_args
                args = kwargs
            solve_result = method(instance, **args)
            if Log.objective in log_param : run_methods_on_instance[method.__name__+"_objective"] = solve_result.best_solution.objective_value
            if Log.runtime in log_param : run_methods_on_instance[method.__name__+"_runtime"] = solve_result.runtime
            if Log.nb_solution in log_param : run_methods_on_instance[method.__name__+"_runtime"] = len(solve_result.all_solutions)
            if Log.status in log_param : run_methods_on_instance[method.__name__+"_runtime"] = solve_result.solve_status.name
        run_methods_on_instances.append(run_methods_on_instance)

    write_excel(fname,run_methods_on_instances)

    return run_methods_on_instances