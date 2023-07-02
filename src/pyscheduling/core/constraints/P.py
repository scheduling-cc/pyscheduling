import BaseConstraint
from constaints_utils import read_2D, write_2D, generate_2D

class P(BaseConstraint):
        
        _name = "P"
        _value = 0

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            P, i = read_2D(instance.n, text_content,starting_index)
            setattr(instance, cls._name, P)
            return i
            
        @classmethod
        def write(cls, instance, file):
            P = getattr(instance, cls._name )
            write_2D(P, file)

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            Pmin = kwargs.get("Pmin")
            Pmax = kwargs.get("Pmax")
            P = generate_2D(instance.n, instance.m, Pmin, Pmax, law)
            setattr(instance, cls._name, P)
            return P
