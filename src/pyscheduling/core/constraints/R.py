import BaseConstraint
from constaints_utils import read_1D, write_1D, generate_1D, flatten_list
import statistics

class R(BaseConstraint):
        
        _name = "R"
        _value = 20

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            R, i = read_1D(text_content,starting_index)
            setattr(instance, cls._name ,R)
            return i

        @classmethod
        def write(cls, instance, file):
            R = getattr(instance, cls._name)
            file.write("\nRelease time\n")
            write_1D(R, file)

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            alpha = kwargs.get("alpha")

            flatten_P = flatten_list(instance.P)
            mean_P = statistics.mean(flatten_P)
            r_max = (instance.n * mean_P * alpha) // instance.m
            r_min = 1

            R = generate_1D(instance.n, r_min, r_max, law)
            setattr(instance, cls._name, R)
            return instance.R