"""Optimal approach for MazeEnv using A* pathfinding algorithm."""

import heapq
from typing import Optional


class GeneratedApproach:
    """Optimal maze solver using A* pathfinding."""

    def __init__(self, action_space, observation_space):
        """Initialize with the environment's gym spaces."""
        self.action_space = action_space
        self.observation_space = observation_space
        self.planned_path: Optional[list[tuple[int, int]]] = None
        self.path_index = 0

        # Action mappings
        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3

        # Direction vectors for each action
        self.action_to_delta = {
            self.UP: (-1, 0),
            self.DOWN: (1, 0),
            self.LEFT: (0, -1),
            self.RIGHT: (0, 1)
        }

    def reset(self, state, info):
        """Called at the start of each episode with the initial state."""
        # Compute optimal path from start to goal using A*
        self.planned_path = self._astar_search(state)
        self.path_index = 0

    def get_action(self, state):
        """Return a valid action for the given state."""
        # If we have a planned path and haven't finished it
        if self.planned_path and self.path_index < len(self.planned_path) - 1:
            current_pos = state.agent
            next_pos = self.planned_path[self.path_index + 1]

            # Find the action that moves us toward the next position
            dr = next_pos[0] - current_pos[0]
            dc = next_pos[1] - current_pos[1]

            # Map delta to action
            for action, (delta_r, delta_c) in self.action_to_delta.items():
                if (dr, dc) == (delta_r, delta_c):
                    self.path_index += 1
                    return action

        # Fallback: if no path or path is complete, try to move toward goal
        return self._greedy_action(state)

    def _astar_search(self, state) -> Optional[list[tuple[int, int]]]:
        """Find optimal path from agent to goal using A* algorithm."""
        start = state.agent
        goal = state.goal
        obstacles = state.obstacles
        height = state.height
        width = state.width

        # Priority queue: (f_score, position, g_score, path)
        heap = [(self._heuristic(start, goal), start, 0, [start])]
        visited = set()

        while heap:
            f_score, current, g_score, path = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)

            # Found goal
            if current == goal:
                return path

            # Explore neighbors
            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                # Check bounds and obstacles
                if (0 <= nr < height and 0 <= nc < width and
                    neighbor not in obstacles and neighbor not in visited):

                    new_g_score = g_score + 1
                    new_f_score = new_g_score + self._heuristic(neighbor, goal)
                    new_path = path + [neighbor]

                    heapq.heappush(heap, (new_f_score, neighbor, new_g_score, new_path))

        return None  # No path found

    def _heuristic(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _greedy_action(self, state) -> int:
        """Greedy action selection toward goal (fallback)."""
        agent_r, agent_c = state.agent
        goal_r, goal_c = state.goal

        # Calculate desired movement direction
        dr = goal_r - agent_r
        dc = goal_c - agent_c

        # Prioritize larger movement direction
        if abs(dr) >= abs(dc):
            if dr > 0:
                return self.DOWN
            elif dr < 0:
                return self.UP
            elif dc > 0:
                return self.RIGHT
            else:
                return self.LEFT
        else:
            if dc > 0:
                return self.RIGHT
            elif dc < 0:
                return self.LEFT
            elif dr > 0:
                return self.DOWN
            else:
                return self.UP