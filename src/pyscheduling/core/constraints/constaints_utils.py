import itertools
import random
from typing import List 
from pyscheduling.Problem import RandomDistrib

# Reading methods
def read_1D(content: List[str], startIndex: int):
    """Read a table from a list of lines extracted from the file of the instance

    Args:
        content (list[str]): lines of the file of the instance
        startIndex (int): Index from where starts the vector

    Returns:
        (list[int],int): (vector, index of the next section of the instance)
    """
    i = startIndex + 1
    vector = list(map(int,content[i].strip().split('\t')))
    return (vector, i+1)


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
def write_1D(array: List[int], file):
    file.write("\t".join(map(str,array)))


def write_2D(matrix: List[List[int]], file):
    nb_lines = len(matrix)
    for i in range(nb_lines):
        BaseConstraints.write_1D(matrix[i], file)
        if i != nb_lines - 1:
            file.write("\n")

# Generation methods
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


def generate_2D(nb_lines: int, nb_columns: int, min_value: int = 1,
                max_value: int = 99, law: RandomDistrib = RandomDistrib.UNIFORM):
    random_matrix = []
    for j in range(nb_lines):
        array_j = BaseConstraints.generate_1D(nb_columns, min_value, max_value, law)
        random_matrix.append(array_j)

    return random_matrix

def generate_3D(nb_matrices: int, nb_lines: int, nb_columns: int,
        min_value: int = 1, max_value: int = 99, law: RandomDistrib = RandomDistrib.UNIFORM):
    
    random_matrices = []
    for i in range(nb_matrices):
        random_matrix = BaseConstraints.generate_2D(nb_lines, nb_columns, min_value, max_value, law)
        random_matrices.append(random_matrix)

    return random_matrices

def flatten_list(array: List) -> List:
    return list(itertools.chain.from_iterable(array)) \
        if isinstance(array[0], list) else array