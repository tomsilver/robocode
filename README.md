# RoboCode

![workflow](https://github.com/tomsilver/robocode/actions/workflows/ci.yml/badge.svg)

Agents for robot physical reasoning.

Work in progress.

## Installation
```bash
git clone https://github.com/tomsilver/robocode.git
cd robocode
bash install.sh
```

### Claude Code CLI setup

The agentic approach requires the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude`). Authenticate via one of:

- **Subscription (free usage):** `claude auth login`
- **API key:** set `ANTHROPIC_API_KEY` in your environment

Optionally set `ROBOCODE_CLAUDE_CMD` to point to a specific `claude` binary (defaults to `claude` on `PATH`).

The `model` parameter in `agentic.yaml` accepts CLI model aliases or full model IDs. Override per-run with e.g. `model=opus` on the command line.

| Alias | Full model ID |
|---|---|
| `sonnet` | `claude-sonnet-4-6` (latest Sonnet, default) |
| `opus` | `claude-opus-4-6` (latest Opus) |
| `haiku` | `claude-haiku-4-5-20251001` (latest Haiku) |

Older model versions can also be used by specifying the full ID:

| Full model ID | Description |
|---|---|
| `claude-sonnet-4-5-20250929` | Claude Sonnet 4.5 |
| `claude-opus-4-5-20251101` | Claude Opus 4.5 |
| `claude-opus-4-1-20250805` | Claude Opus 4.1 |
| `claude-sonnet-4-20250514` | Claude Sonnet 4 |
| `claude-opus-4-20250514` | Claude Opus 4 |
| `claude-3-7-sonnet-20250219` | Claude 3.7 Sonnet |

See [Anthropic models overview](https://platform.claude.com/docs/en/about-claude/models/overview) for the full list.

## Environments

All environments are available as Hydra configs via `environment=<config_name>`.

### Maze (discrete)

| Config | Description |
|---|---|
| `small_maze` | Small grid maze |
| `large_maze` | Large grid maze |

### 2D Kinematic (continuous, kinder geom2d)

| Config | Kinder ID | Difficulty |
|---|---|---|
| `motion2d_easy` | `kinder/Motion2D-p0-v0` | Easy (0 passages) |
| `motion2d_medium` | `kinder/Motion2D-p1-v0` | Medium (1 passage) |
| `motion2d_hard` | `kinder/Motion2D-p3-v0` | Hard (3 passages) |
| `obstruction2d_easy` | `kinder/Obstruction2D-o0-v0` | Easy (0 obstructions) |
| `obstruction2d_medium` | `kinder/Obstruction2D-o2-v0` | Medium (2 obstructions) |
| `obstruction2d_hard` | `kinder/Obstruction2D-o4-v0` | Hard (4 obstructions) |
| `clutteredretrieval2d_easy` | `kinder/ClutteredRetrieval2D-o1-v0` | Easy (1 obstruction) |
| `clutteredretrieval2d_medium` | `kinder/ClutteredRetrieval2D-o10-v0` | Medium (10 obstructions) |
| `clutteredretrieval2d_hard` | `kinder/ClutteredRetrieval2D-o25-v0` | Hard (25 obstructions) |
| `clutteredstorage2d_easy` | `kinder/ClutteredStorage2D-b1-v0` | Easy (1 block) |
| `clutteredstorage2d_medium` | `kinder/ClutteredStorage2D-b3-v0` | Medium (3 blocks) |
| `clutteredstorage2d_hard` | `kinder/ClutteredStorage2D-b7-v0` | Hard (7 blocks) |
| `stickbutton2d_easy` | `kinder/StickButton2D-b1-v0` | Easy (1 button) |
| `stickbutton2d_medium` | `kinder/StickButton2D-b3-v0` | Medium (3 buttons) |
| `stickbutton2d_hard` | `kinder/StickButton2D-b5-v0` | Hard (5 buttons) |
| `pushpullhook2d` | `kinder/PushPullHook2D-v0` | Single variant |

### 3D Kinematic (continuous, kinder geom3d)

| Config | Kinder ID | Difficulty |
|---|---|---|
| `motion3d` | `kinder/Motion3D-v0` | Single variant |
| `obstruction3d_easy` | `kinder/Obstruction3D-o0-v0` | Easy (0 obstructions) |
| `obstruction3d_medium` | `kinder/Obstruction3D-o2-v0` | Medium (2 obstructions) |
| `obstruction3d_hard` | `kinder/Obstruction3D-o4-v0` | Hard (4 obstructions) |
| `shelf3d_easy` | `kinder/Shelf3D-o1-v0` | Easy (1 cube) |
| `shelf3d_medium` | `kinder/Shelf3D-o3-v0` | Medium (3 cubes) |
| `shelf3d_hard` | `kinder/Shelf3D-o5-v0` | Hard (5 cubes) |
| `transport3d_easy` | `kinder/Transport3D-o1-v0` | Easy (1 cube) |
| `transport3d_hard` | `kinder/Transport3D-o2-v0` | Hard (2 cubes) |
| `packing3d_easy` | `kinder/Packing3D-p1-v0` | Easy (1 part) |
| `packing3d_medium` | `kinder/Packing3D-p2-v0` | Medium (2 parts) |
| `packing3d_hard` | `kinder/Packing3D-p3-v0` | Hard (3 parts) |

## Sandbox

The agent runs inside a Docker container (`robocode-sandbox`) that provides full filesystem isolation, a restricted network, and a pre-built Python environment.

### Security model

| Layer | Mechanism |
|---|---|
| Filesystem | Docker bind-mount: agent can only write to `/sandbox` (the run's output dir) |
| Network | `init-firewall.sh` whitelists `api.anthropic.com`, GitHub IPs, and Claude telemetry; blocks everything else via iptables |
| Write hook | `PreToolUse` hook in `.claude/settings.json` double-checks Write/Edit paths stay inside `/sandbox` |

### What the agent sees

| Path | Contents |
|---|---|
| `/sandbox/` | Working directory — agent writes `approach.py`, test scripts, etc. here |
| `/sandbox/primitives/` | Source files from `src/robocode/primitives/` (read reference) |
| `/robocode/.venv/bin/python` | Python 3.11 with all robocode dependencies pre-installed |
| `/robocode/prpl-mono/` | Third-party packages, bind-mounted read-only from the host submodule |

### Start docker

#### Mac OS

Simply open the Docker Desktop application.
Look for the status indicator in the bottom-left corner of the GUI; it should say "Docker Engine Running".

#### Linux
```
sudo systemctl start docker
sudo systemctl enable docker
```

### Building the image

Build once from the repo root (rebuild when `pyproject.toml` / `uv.lock` change; not needed for `prpl-mono` code changes):

```bash
bash docker/build.sh
```

### Using the OS-level sandbox (legacy)

The original macOS Seatbelt / Linux bubblewrap sandbox is still available (`use_docker: false` in `agentic.yaml`) but has a known limitation: it restricts filesystem *writes* but allows *reads* of the entire host filesystem.

Red team the sandbox:
```bash
python integration_tests/red_team_sandbox.py           # OS-level
python integration_tests/red_team_sandbox.py --docker  # Docker
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

The `agentic` approach launches a Claude agent during `train()`. The agent reads the environment source code, figures out the state/action space and dynamics, and writes a `GeneratedApproach` class that is used at evaluation time. The agent can also write and run test scripts against the real environment to verify its solution before committing.

By default the agent runs in the Docker sandbox (requires `bash docker/build.sh` once):
```bash
python experiments/run_experiment.py approach=agentic environment=motion2d_easy
```

To use the legacy OS-level sandbox instead:
```bash
python experiments/run_experiment.py approach=agentic environment=small_maze approach.use_docker=false
```

To skip re-generation and load a previously generated approach:
```bash
python experiments/run_experiment.py approach=agentic environment=small_maze \
    approach.load_dir=outputs/2026-02-16/16-00-41
```

Parallel sweeps each get their own container (named `robocode-sandbox-<uuid>`), so multiple runs never interfere:
```bash
python experiments/run_experiment.py -m seed=0,1,2 environment=small_maze,large_maze approach=agentic
```

The generated `approach.py` and full agent log are saved under `sandbox/` in the run's output directory (e.g. `outputs/2026-02-16/16-00-41/sandbox/`).

#### Example: small_maze

On `small_maze`, the agent independently discovered A* pathfinding and achieved a **100% solve rate with optimal path lengths** (mean 2.3 steps across 10 episodes):

```json
{
  "mean_eval_reward": -2.3,
  "mean_eval_steps": 2.3,
  "solve_rate": 1.0,
  "num_eval_tasks": 10
}
```

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
