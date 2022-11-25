# read version from installed package
from importlib.metadata import version
__version__ = version("pyscheduling")

__all__ = ["SingleMachine", "SM_Methods", "risijwiCi", "risijwiFi", "wiCi", "riwiCi", "wiTi", "riwiTi", "sijwiTi", "risijwiTi", "sijCmax", "risijCmax"]