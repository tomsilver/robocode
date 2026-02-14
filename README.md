# RoboCode

![workflow](https://github.com/tomsilver/robocode/actions/workflows/ci.yml/badge.svg)

Agents for robot physical reasoning.

Work in progress.

## Experiments

Run an experiment:
```bash
python experiments/run_experiment.py approach=random environment=small_maze seed=0
```

Run a sweep over multiple seeds and environments:
```bash
python experiments/run_experiment.py -m seed=0,1,2 environment=small_maze,large_maze approach=random
```

Analyze results from one or more runs:
```bash
python experiments/analyze_results.py multirun/
```
