import BaseConstraint
from constaints_utils import read_1D, write_1D, generate_1D

class W(BaseConstraint):

        _name = "W"
        _value = 10

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            W, i = read_1D(text_content,starting_index)
            setattr(instance, cls._name, W)
            return i
            
        @classmethod
        def write(cls, instance, file):
            W = getattr(instance, cls._name)
            file.write("\nWeights\n")
            write_1D(W, file)

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            Wmin = kwargs.get("Wmin")
            Wmax = kwargs.get("Wmax")
            W = generate_1D(instance.n, min_value=Wmin, max_value=Wmax, law=law)
            setattr(instance, cls._name, W)
            return W
        
