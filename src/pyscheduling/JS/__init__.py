# read version from installed package
from importlib.metadata import version
__version__ = version("pyscheduling")

__all__ = ["JobShop", "JmCmax", "JmridiSijkwiTi", "JmriSijkCmax", "JmriSijkwiCi",
           "JmriSijkwiFi", "JmriwiTi", "JS_methods"]