from typing import List
from pyscheduling.BaseConstraints import BaseConstraints

class Constraints(BaseConstraints):

    class P(BaseConstraints.P):

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            P, i = Constraints.read_1D(instance.n, text_content,starting_index)
            setattr(instance, cls._name, P)
            return i
        
        @classmethod
        def write(cls, instance, file):
            P = getattr(instance, cls._name )
            for i in range(instance.n):
                file.write(str(P[i])+"\t")

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            Pmin = kwargs.get("Pmin")
            Pmax = kwargs.get("Pmax")
            P = Constraints.generate_1D(instance.n, Pmin, Pmax, law)
            setattr(instance, cls._name, P)
            return P
        
    class S(BaseConstraints.S):

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            S, i = Constraints.read_2D(instance.n, text_content, starting_index)
            setattr(instance, cls._name, S)
            return i

        @classmethod
        def write(cls, instance, file):
            S = getattr(instance,"S")
            file.write("\nSSD\n")
            for i in range(instance.n):
                for j in range(instance.n):
                    file.write(str(S[i][j])+"\t")
                file.write("\n")
    
        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            gamma = kwargs.get("gamma")

            flatten_P = Constraints.flatten_list(instance.P)
            Pmin = min(flatten_P)
            Pmax = max(flatten_P)
            # Limit values
            Smin = int(gamma * Pmin)
            Smax = int(gamma * Pmax)

            S = Constraints.generate_2D(instance.n, instance.n,
                                        Smin, Smax, law)
            setattr(instance, cls._name, S)
            return instance.S