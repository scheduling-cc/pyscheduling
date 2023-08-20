# read version from installed package
from importlib.metadata import version
__version__ = version("pyscheduling")

__all__ = ["base_solvers", "listeners", "BaseConstraints", "benchmark"]

from . import base_solvers
from . import listeners
from . import BaseConstraints
from . import benchmark
