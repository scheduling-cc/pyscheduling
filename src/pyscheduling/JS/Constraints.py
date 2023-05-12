import random
import statistics
from typing import List

from pyscheduling.core.BaseConstraints import BaseConstraints


class Constraints(BaseConstraints):

    class P(BaseConstraints.P):

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            P = []  
            i = starting_index + 1
            for _ in range(instance.n):
                ligne = text_content[i].strip().split('\t')
                P_k = [(int(ligne[j-1]),int(ligne[j])) for j in range(1, len(ligne), 2)]
                P.append(P_k)
                i += 1
            
            setattr(instance, cls._name, P)
            return i
            
        @classmethod
        def write(cls, instance, file):
            for i, job in enumerate(instance.P):
                file.write( "\t".join(f'{m_id}\t{p_oper}' for m_id, p_oper in job) )
                if i != instance.n - 1:
                    file.write("\n")

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            Pmin = kwargs.get("Pmin")
            Pmax = kwargs.get("Pmax")
            P = []
            visited_machine = list(range(instance.m))
            for j in range(instance.n):
                Pj = []
                nb_operation_j = random.randint(1, instance.m)
                for _ in range(nb_operation_j):
                    machine_id = random.randint(0, instance.m-1)
                    while(machine_id in [i[0] for i in Pj]) : machine_id = random.randint(0, instance.m-1) # checks recirculation
                    if machine_id in visited_machine: visited_machine.remove(machine_id)
                    if law.name == "UNIFORM":  # Generate uniformly
                        n = int(random.uniform(Pmin, Pmax))
                    elif law.name == "NORMAL":  # Use normal law
                        n = -1
                        while n < Pmin or n > Pmax:
                            value = random.gauss(0, 1)
                            n = int(abs(Pmin+Pmax*value))
                    Pj.append((machine_id,n))
                P.append(Pj)
            #If there are some unused machine by any operation
            if len(visited_machine) > 0:
                for job_list_id in range(instance.n):
                    for machine_id in range(job_list_id,len(visited_machine),instance.n):
                        if law.name == "UNIFORM":  # Generate uniformly
                            n = int(random.uniform(Pmin, Pmax))
                        elif law.name == "NORMAL":  # Use normal law
                            n = -1
                            while n < Pmin or n > Pmax:
                                value = random.gauss(0, 1)
                                n = int(abs(Pmin+Pmax*value))
                        P[job_list_id].append((machine_id,n))
            
            setattr(instance, cls._name, P)
            return P

    class R(BaseConstraints.R):

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            alpha = kwargs.get("alpha")
            
            p_values = [oper[1] for job in instance.P for oper in job]
            flatten_P = BaseConstraints.flatten_list(p_values)
            mean_P = statistics.mean(flatten_P)
            r_max = (instance.n * mean_P * alpha) // instance.m
            r_min = 1

            R = BaseConstraints.generate_1D(instance.n, r_min, r_max, law)
            setattr(instance, cls._name, R)
            return instance.R
        
    class D(BaseConstraints.D):

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            due_time_factor = kwargs.get("due_time_factor")

            p_values = [oper[1] for job in instance.P for oper in job]
            flatten_P = BaseConstraints.flatten_list(p_values)
            Pmin = min(flatten_P)
            Pmax = max(flatten_P)
            mean_P = statistics.mean(flatten_P)
            mean_Ci = (mean_P * instance.n) // instance.m
            max_R = max(instance.R) if hasattr(instance,"R") else 0
            # Limit values for d
            min_D = max_R + Pmin
            max_D = int((max_R + mean_Ci) * due_time_factor)

            D = BaseConstraints.generate_1D(instance.n, min_D, max_D, law)
            setattr(instance, cls._name, D)
            return D
        
    class S(BaseConstraints.S):
        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            gamma = kwargs.get("gamma")

            p_values = [oper[1] for job in instance.P for oper in job]
            flatten_P = BaseConstraints.flatten_list(p_values)
            Pmin = min(flatten_P)
            Pmax = max(flatten_P)
            # Limit values
            Smin = int(gamma * Pmin)
            Smax = int(gamma * Pmax)

            S = BaseConstraints.generate_3D(instance.m, instance.n, instance.n,
                                        Smin, Smax, law)
            setattr(instance, cls._name, S)
            return instance.S
    
