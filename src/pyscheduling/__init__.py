# read version from installed package
from importlib.metadata import version
__version__ = version("pyscheduling")

__all__ = ["SMSP", "PMSP", "FS", "JS", "Problem", "core"]

from . import SMSP
from . import PMSP
from . import FS
from . import JS
from . import core
