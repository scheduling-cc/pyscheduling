import os
import sys
from pathlib import Path
import time
import csv

def write_excel(fname : Path, result):
    """ Wrapper method to bypass pandas dependancy, mainly used in opale server
    Args:
        fname (str): path to the result excel file, without the .extension
        result_dict (dict): (key, value) pairs of the results 
    """
    result_list = result
    if type(result) is dict:
        result_list = [result]
    keys = result_list[0].keys()

    with open(fname + ".csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(result_list)

