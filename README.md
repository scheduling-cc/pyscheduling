# pyscheduling_cc

THE python package to solve scheduling problems

## Installation

```bash
$ pip install pyscheduling_cc
```

## Usage

```python
import pyscheduling_cc as pyscheduling
import pyscheduling_cc.ParallelMachines as pmsp
import pyscheduling_cc.RmSijkCmax as pmsp_sijk

instance = pmsp_sijk.RmSijkCmax_Instance.generate_random(150,10)
solver = pyscheduling.Problem.Solver(pmsp_sijk.Heuristics.constructive)
solve_result = solver.solve(instance)
LSOperator = pmsp.PM_LocalSearch()
LSOperator.improve(solve_result.best_solution)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pyscheduling_cc` was created by the scheduling-cc organization. It is licensed under the terms of the MIT license.
