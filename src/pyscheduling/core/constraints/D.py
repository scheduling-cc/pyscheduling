import BaseConstraint
from constaints_utils import read_1D, write_1D, generate_1D, flatten_list
import statistics

class D(BaseConstraint):
        
        _name = "D"
        _value = 30

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            D, i = read_1D(text_content,starting_index)
            setattr(instance, cls._name ,D)
            return i

        @classmethod
        def write(cls, instance, file):
            D = getattr(instance, cls._name )
            file.write("\nDue time\n")
            write_1D(D, file)

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            due_time_factor = kwargs.get("due_time_factor")

            flatten_P = flatten_list(instance.P)
            Pmin = min(flatten_P)
            Pmax = max(flatten_P)
            mean_P = statistics.mean(flatten_P)
            mean_Ci = (mean_P * instance.n) // instance.m
            max_R = max(instance.R) if hasattr(instance,"R") else 0
            # Limit values for d
            min_D = max_R + Pmin
            max_D = int((max_R + mean_Ci) * due_time_factor)

            D = generate_1D(instance.n, min_D, max_D, law)
            setattr(instance, cls._name, D)
            return D