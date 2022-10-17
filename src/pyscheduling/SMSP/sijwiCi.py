import sys
from dataclasses import dataclass, field
from random import randint
from pathlib import Path
from time import perf_counter

import pyscheduling.Problem as RootProblem
from pyscheduling.Problem import Solver, Constraints, Objective
import pyscheduling.SMSP.SingleMachine as SingleMachine
from pyscheduling.SMSP.SingleMachine import single_instance
from pyscheduling.SMSP.SM_Methods import ExactSolvers


@single_instance([Constraints.W, Constraints.S], Objective.wiCi)
class sijwiCi_Instance(SingleMachine.SingleInstance):
    pass
