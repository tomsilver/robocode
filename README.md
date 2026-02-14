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

### Agentic approach

The `agentic` approach launches a sandboxed Claude agent during training. The agent reads the environment source code, figures out the state/action space and dynamics, and writes an approach class that is used at evaluation time. The agent can also write and run test scripts to verify its solution.

```bash
python experiments/run_experiment.py approach=agentic environment=small_maze
```

On `small_maze`, the agent independently discovered A* pathfinding and achieved a **100% solve rate with optimal path lengths** (mean 2.3 steps across 10 episodes):

```json
{
  "mean_eval_reward": -2.3,
  "mean_eval_steps": 2.3,
  "solve_rate": 1.0,
  "num_eval_tasks": 10
}
```

The generated approach class and full agent log are saved in the run's output directory under `sandbox/`.

<details>
<summary>Generated <code>approach.py</code> (A* pathfinding)</summary>

```python
"""Optimal approach for MazeEnv using A* pathfinding algorithm."""

import heapq
from typing import Optional


class GeneratedApproach:
    """Optimal maze solver using A* pathfinding."""

    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.planned_path: Optional[list[tuple[int, int]]] = None
        self.path_index = 0

        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3

        self.action_to_delta = {
            self.UP: (-1, 0),
            self.DOWN: (1, 0),
            self.LEFT: (0, -1),
            self.RIGHT: (0, 1)
        }

    def reset(self, state, info):
        self.planned_path = self._astar_search(state)
        self.path_index = 0

    def get_action(self, state):
        if self.planned_path and self.path_index < len(self.planned_path) - 1:
            next_pos = self.planned_path[self.path_index + 1]
            dr = next_pos[0] - state.agent[0]
            dc = next_pos[1] - state.agent[1]
            for action, (delta_r, delta_c) in self.action_to_delta.items():
                if (dr, dc) == (delta_r, delta_c):
                    self.path_index += 1
                    return action
        return self._greedy_action(state)

    def _astar_search(self, state) -> Optional[list[tuple[int, int]]]:
        start, goal = state.agent, state.goal
        heap = [(self._heuristic(start, goal), start, 0, [start])]
        visited = set()
        while heap:
            _, current, g_score, path = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                return path
            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if (0 <= nr < state.height and 0 <= nc < state.width
                    and neighbor not in state.obstacles
                    and neighbor not in visited):
                    new_g = g_score + 1
                    heapq.heappush(heap, (
                        new_g + self._heuristic(neighbor, goal),
                        neighbor, new_g, path + [neighbor]
                    ))
        return None

    def _heuristic(self, pos1, pos2) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _greedy_action(self, state) -> int:
        dr = state.goal[0] - state.agent[0]
        dc = state.goal[1] - state.agent[1]
        if abs(dr) >= abs(dc):
            return self.DOWN if dr > 0 else self.UP
        return self.RIGHT if dc > 0 else self.LEFT
```
</details>

<details>
<summary>Agent log (19 turns, $0.36)</summary>

The agent's workflow:
1. Explored the sandbox directory and read `maze_env.py` and `base_env.py`
2. Analyzed the state type (`_MazeState` with `agent`, `goal`, `obstacles`, `height`, `width`), action space (4 directional moves), and reward structure (-1 per step)
3. Identified this as a shortest-path problem and implemented A* with Manhattan distance heuristic
4. Wrote and ran three test scripts verifying correctness on simple mazes, complex mazes with obstacles, impossible mazes (graceful fallback), and optimality of path lengths
5. Cleaned up test files and produced the final `approach.py`

```
18:32:05 Agent: I'll analyze the environment source files and create an optimal approach.py.
18:32:06 Tool call: Glob({'pattern': '**/*.py'})
18:32:08 Agent: Let me explore the main environment files to understand the structure:
18:32:09 Tool call: Read({'.../maze_env.py'})
18:32:09 Tool call: Read({'.../base_env.py'})
18:32:11 Agent: [Analyzed state/action space, identified A* as optimal strategy]
18:32:40 Tool call: Write({'.../approach.py'})
18:32:43 Agent: Now let me create a simple test script to verify the approach works:
18:33:00 Tool call: Write({'.../test_approach.py'})
18:33:03 Tool call: Bash({'command': 'python test_approach.py'})
18:33:07 Agent: Excellent! Testing on a more complex maze...
18:33:22 Tool call: Write({'.../test_complex_maze.py'})
18:33:26 Tool call: Bash({'command': 'python test_complex_maze.py'})
18:33:31 Agent: A* correctly found optimal 12-step path around all obstacles.
18:33:39 Tool call: Write({'.../test_real_state.py'})
18:33:54 Tool call: Bash({'command': 'python test_real_state.py'})
18:33:58 Agent: Works with actual _MazeState class.
18:34:09 Tool call: Write({'.../test_optimality.py'})
18:34:11 Tool call: Bash({'command': 'python test_optimality.py'})
18:34:13 Agent: All test cases produce optimal solutions.
18:34:14 Tool call: Bash({'command': 'rm test_*.py'})
18:34:25 Session done: turns=19, cost=$0.3628055, error=False
```
</details>
