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

def run_solver_instance(instances : list[Problem.Instance],methods : list[object],kwargs={}):
    instances_names = [instance.name for instance in instances]
    #methods_names = [method.__name__ for method in methods]
    run_methods_on_instances = []
    for method in methods :
        run_method_on_instances = {instance_name: (-1,-1) for instance_name in instances_names}
        for instance in instances :
            solve_result = method(instance, **kwargs)
            run_method_on_instances[instance.name] = (solve_result.best_solution.objective_value,solve_result.runtime)
        run_methods_on_instances.append(run_method_on_instances)

    write_excel("benchmark_results.csv",run_methods_on_instances)

    return run_methods_on_instances