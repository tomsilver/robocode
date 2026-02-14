# RoboCode

![workflow](https://github.com/tomsilver/robocode/actions/workflows/ci.yml/badge.svg)

Agents for robot physical reasoning.

Work in progress.

## Sandbox

The `robocode.sandbox` module runs a Claude agent in a restricted working directory. The agent can use Bash, Read, Write, Edit, Glob, and Grep tools, but file tools are restricted to the sandbox directory via a PreToolUse hook and Bash is sandboxed at the OS level (macOS Seatbelt / Linux bubblewrap).

**Known limitation:** The OS-level sandbox restricts filesystem *writes* but allows *reads* of the entire filesystem. Bash commands like `cat /etc/passwd` or Python's `open()` can read files outside the sandbox. This will be addressed by transitioning to Docker-based sandboxing. We may also move from the Claude Agent SDK to running Claude directly (e.g., via the Anthropic API with tool use), which would give us full control over tool execution rather than relying on the SDK's built-in tool dispatch.

Red team the sandbox:
```bash
python integration_tests/red_team_sandbox.py
```

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
