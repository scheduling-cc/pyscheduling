# read version from installed package
from importlib.metadata import version
__version__ = version("pyscheduling")

__all__ = ["FlowShop", "FS_Methods", "FmCmax", "FmSijkCmax"]