import BaseConstraint
from constaints_utils import read_3D, flatten_list, generate_3D, write_2D


class S(BaseConstraint):
        
        _name = "S"
        _value = 40

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            S, i = read_3D(instance.m, instance.n, text_content, starting_index)
            setattr(instance, cls._name, S)
            return i

        @classmethod
        def write(cls, instance, file):
            S = getattr(instance, cls._name)
            file.write("\nSSD")
            for i in range(instance.m):
                file.write("\nM"+str(i)+"\n")
                write_2D(S[i], file)
    
        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            gamma = kwargs.get("gamma")

            flatten_P = flatten_list(instance.P)
            Pmin = min(flatten_P)
            Pmax = max(flatten_P)
            # Limit values
            Smin = int(gamma * Pmin)
            Smax = int(gamma * Pmax)

            S = generate_3D(instance.m, instance.n, instance.n,
                                        Smin, Smax, law)
            setattr(instance, cls._name, S)
            return instance.S