# read version from installed package
from importlib.metadata import version
__version__ = version("pyscheduling_cc")

__all__ = ["JobShop", "JmCmax"]