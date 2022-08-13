import os
import sys
from pathlib import Path
import time
import csv

import pyscheduling_cc.Problem as Problem

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

def run_solver_instance(instances : list[Problem.Instance],methods_args : list[object],kwargs={}):
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
            run_methods_on_instance[method.__name__+"_objective"] = solve_result.best_solution.objective_value
            run_methods_on_instance[method.__name__+"_runtime"] = solve_result.runtime
        run_methods_on_instances.append(run_methods_on_instance)

    write_excel("benchmark_results.csv",run_methods_on_instances)

    return run_methods_on_instances