# pyscheduling

THE python package to solve scheduling problems

## Installation

```bash
$ pip install pyscheduling
```

## Usage

```python
import pyscheduling.SMSP.interface as sm

problem = sm.Problem()
problem.add_constraints([sm.Constraints.W,sm.Constraints.D])
problem.set_objective(sm.Objective.wiTi)
problem.generate_random(jobs_number=20,Wmax=10)
solution = problem.solve(problem.heuristics["ACT"])
print(solution)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pyscheduling` was created by the scheduling-cc organization. CC-BY-NC-ND-4.0.