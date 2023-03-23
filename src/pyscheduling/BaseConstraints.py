import itertools
import random
import statistics
from typing import List

from pyscheduling.Problem import RandomDistrib


class BaseConstraints():

    # Reading methods
    @staticmethod
    def read_1D(content: List[str], startIndex: int):
        """Read a table from a list of lines extracted from the file of the instance

        Args:
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the vector

        Returns:
           (list[int],int): (vector, index of the next section of the instance)
        """
        i = startIndex + 1
        line = content[i].strip().split('\t')
        vector = []  # Table : Processing time of job i
        for j in line:
            vector.append(int(j))
        return (vector, i+1)

    @staticmethod
    def read_2D(dimension_i : int, content: List[str], startIndex: int):
        """Read a matrix from a list of lines extracted from the file of the instance

        Args:
            dimension_i (int): number of lines of the matrix, usually number of jobs 'n'.
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the matrix

        Returns:
           (list[list[int]],int): (Matrix, index of the next section of the instance)
        """
        i = startIndex
        Matrix = []  # Matrix S_ijk : Setup time between jobs j and k
        i += 1  # Skip SSD
        for k in range(dimension_i):
            line = content[i].strip().split('\t')
            Matrix_i = [int(val_str) for val_str in line]
            Matrix.append(Matrix_i)
            i += 1
        return (Matrix, startIndex+1+dimension_i)

    @staticmethod
    def read_3D(dimension_i : int, dimension_j : int, content: List[str], startIndex: int):
        """Read the table of matrices from a list of lines extracted from the file of the instance

        Args:
            dimension_i (int): Dimension of the table, usually number of machines 'm'.
            dimension_j (int): Dimension of the matrix, usually number of jobs 'n'.
            content (list[str]): lines of the file of the instance
            startIndex (int): Index from where starts the table of matrices

        Returns:
           (list[list[list[int]]],int): (Table of matrices, index of the next section of the instance)
        """
        i = startIndex
        S = []  # Table of Matrix S_ijk : Setup time between jobs j and k on machine i
        i += 1  # Skip SSD
        endIndex = startIndex+1+dimension_j*dimension_i+dimension_i
        while i != endIndex:
            i = i+1  # Skip Mk
            Si = []
            for k in range(dimension_j):
                ligne = content[i].strip().split('\t')
                Sij = [int(val_str) for val_str in ligne]
                Si.append(Sij)
                i += 1
            S.append(Si)
        return (S, i)

    # Output methods
    @staticmethod
    def write_1D(array: List[int], file):
        file.write("\t".join(map(str,array)))

    @staticmethod
    def write_2D(matrix: List[List[int]], file):
        nb_lines = len(matrix)
        for i in range(nb_lines):
            BaseConstraints.write_1D(matrix[i], file)
            if i != nb_lines - 1:
                file.write("\n")

    # Generation methods
    @staticmethod
    def generate_1D(nb_values: int, min_value: int = 1, max_value: int = 99,
                    law : RandomDistrib = RandomDistrib.UNIFORM):
        random_array = []
        for j in range(nb_values):
            if law == RandomDistrib.UNIFORM:  # Generate uniformly
                n = int(random.randint(min_value, max_value))
            elif law == RandomDistrib.NORMAL:  # Use normal law
                n = -1
                while n < min_value or n > max_value:
                    value = random.gauss(0, 1)
                    n = int(abs(min_value+max_value*value))
            random_array.append(n)

        return random_array

    @staticmethod
    def generate_2D(nb_lines: int, nb_columns: int, min_value: int = 1,
                    max_value: int = 99, law: RandomDistrib = RandomDistrib.UNIFORM):
        random_matrix = []
        for j in range(nb_lines):
            array_j = BaseConstraints.generate_1D(nb_columns, min_value, max_value, law)
            random_matrix.append(array_j)

        return random_matrix

    @staticmethod
    def generate_3D(nb_matrices: int, nb_lines: int, nb_columns: int,
            min_value: int = 1, max_value: int = 99, law: RandomDistrib = RandomDistrib.UNIFORM):
        
        random_matrices = []
        for i in range(nb_matrices):
            random_matrix = BaseConstraints.generate_2D(nb_lines, nb_columns, min_value, max_value, law)
            random_matrices.append(random_matrix)

        return random_matrices

    @staticmethod
    def flatten_list(array: List) -> List:
        return list(itertools.chain.from_iterable(array)) \
            if isinstance(array[0], list) else array


    # Constraints classes
    class BaseConstraint():
        
        @classmethod
        def create(cls, instance,var):
            setattr(instance, cls._name ,var) if var is not None else setattr(instance, cls._name, list())

        def __lt__(self, other):
            return self._value < other._value

    class P(BaseConstraint):
        
        _name = "P"
        _value = 0

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            P, i = BaseConstraints.read_2D(instance.n, text_content,starting_index)
            setattr(instance, cls._name, P)
            return i
            
        @classmethod
        def write(cls, instance, file):
            P = getattr(instance, cls._name )
            BaseConstraints.write_2D(P, file)

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            Pmin = kwargs.get("Pmin")
            Pmax = kwargs.get("Pmax")
            P = BaseConstraints.generate_2D(instance.n, instance.m, Pmin, Pmax, law)
            setattr(instance, cls._name, P)
            return P

    class W(BaseConstraint):

        _name = "W"
        _value = 10

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            W, i = BaseConstraints.read_1D(text_content,starting_index)
            setattr(instance, cls._name, W)
            return i
            
        @classmethod
        def write(cls, instance, file):
            W = getattr(instance, cls._name)
            file.write("\nWeights\n")
            BaseConstraints.write_1D(W, file)

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            Wmin = kwargs.get("Wmin")
            Wmax = kwargs.get("Wmax")
            W = BaseConstraints.generate_1D(instance.n, min_value=Wmin, max_value=Wmax, law=law)
            setattr(instance, cls._name, W)
            return W

    class R(BaseConstraint):
        
        _name = "R"
        _value = 20

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            R, i = BaseConstraints.read_1D(text_content,starting_index)
            setattr(instance, cls._name ,R)
            return i

        @classmethod
        def write(cls, instance, file):
            R = getattr(instance, cls._name)
            file.write("\nRelease time\n")
            BaseConstraints.write_1D(R, file)

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            alpha = kwargs.get("alpha")

            flatten_P = BaseConstraints.flatten_list(instance.P)
            mean_P = statistics.mean(flatten_P)
            r_max = (instance.n * mean_P * alpha) // instance.m
            r_min = 1

            R = BaseConstraints.generate_1D(instance.n, r_min, r_max, law)
            setattr(instance, cls._name, R)
            return instance.R
    
    class D(BaseConstraint):
        
        _name = "D"
        _value = 30

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            D, i = BaseConstraints.read_1D(text_content,starting_index)
            setattr(instance, cls._name ,D)
            return i

        @classmethod
        def write(cls, instance, file):
            D = getattr(instance, cls._name )
            file.write("\nDue time\n")
            BaseConstraints.write_1D(D, file)

        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            due_time_factor = kwargs.get("due_time_factor")

            flatten_P = BaseConstraints.flatten_list(instance.P)
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
        
    class S(BaseConstraint):
        
        _name = "S"
        _value = 40

        @classmethod
        def read(cls, instance, text_content : List[str], starting_index : int):
            S, i = BaseConstraints.read_3D(instance.m, instance.n, text_content, starting_index)
            setattr(instance, cls._name, S)
            return i

        @classmethod
        def write(cls, instance, file):
            S = getattr(instance, cls._name)
            file.write("\nSSD")
            for i in range(instance.m):
                file.write("\nM"+str(i)+"\n")
                BaseConstraints.write_2D(S[i], file)
    
        @classmethod
        def generate_random(cls, instance,**kwargs):
            law = kwargs.get("law")
            gamma = kwargs.get("gamma")

            flatten_P = BaseConstraints.flatten_list(instance.P)
            Pmin = min(flatten_P)
            Pmax = max(flatten_P)
            # Limit values
            Smin = int(gamma * Pmin)
            Smax = int(gamma * Pmax)

            S = BaseConstraints.generate_3D(instance.m, instance.n, instance.n,
                                        Smin, Smax, law)
            setattr(instance, cls._name, S)
            return instance.S