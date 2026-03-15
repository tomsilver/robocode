"""Approach for PushPullHook2D that composes pick and push skills.

Uses GroundPickController to grasp the hook, then GroundPushController
to push the movable button towards the target button.
"""

import numpy as np
from kinder.envs.kinematic2d.utils import CRVRobotActionSpace, get_suctioned_objects

from robocode.skills.pushpullhook2d.pick_skill import GroundPickController
from robocode.skills.pushpullhook2d.push_skill import GroundPushController
from robocode.skills.utils import TrajectorySamplingFailure


class GeneratedApproach:
    """Two-phase approach: pick the hook, then push the button."""

    def __init__(self, action_space, observation_space, skills=None,
                    initial_constant_state=None):
        self._action_space = action_space
        self._rng = np.random.default_rng(0)

        # Skill classes — use provided skills dict or fall back to imports.
        if skills and "GroundPickController" in skills:
            self._pick_cls = skills["GroundPickController"]
        else:
            self._pick_cls = GroundPickController
        if skills and "GroundPushController" in skills:
            self._push_cls = skills["GroundPushController"]
        else:
            self._push_cls = GroundPushController

        # Runtime state.
        self._phase = "pick"  # "pick" -> "push" -> "done"
        self._init_constant_state = initial_constant_state
        self._action_queue: list = []
        self._pick_attempts = 0
        self._push_attempts = 0
        self._max_skill_attempts = 10

    def reset(self, state, info):
        self._phase = "pick"
        self._action_queue = []
        self._pick_attempts = 0
        self._push_attempts = 0
        self._rng = np.random.default_rng(0)
        self._state = state

    def get_action(self, state):
        self._state = state

        # If we have queued actions from the current skill, execute them.
        if self._action_queue:
            return self._action_queue.pop(0)

        if self._phase == "pick":
            return self._do_pick(state)
        if self._phase == "push":
            return self._do_push(state)

        # "done" phase — just idle.
        return np.zeros(5, dtype=np.float32)

    def _do_pick(self, state):
        """Try to execute the pick skill."""
        obj_map = {o.name: o for o in state}
        robot = obj_map["robot"]
        hook = obj_map["hook"]

        # Check if already grasped (e.g. from a previous attempt).
        suctioned = get_suctioned_objects(state, robot)
        if any(o.name == "hook" for o, _ in suctioned):
            self._phase = "push"
            return self._do_push(state)

        if self._pick_attempts >= self._max_skill_attempts:
            self._phase = "done"
            return np.zeros(5, dtype=np.float32)

        assert isinstance(self._action_space, CRVRobotActionSpace)
        controller = self._pick_cls(
            objects=[robot, hook],
            action_space=self._action_space,
            init_constant_state=self._init_constant_state,
        )
        params = controller.sample_parameters(state, self._rng)
        self._pick_attempts += 1

        try:
            controller.reset(state, params)
            actions = []
            while not controller.terminated():
                actions.append(controller.step())
            self._action_queue = actions
            if self._action_queue:
                return self._action_queue.pop(0)
        except TrajectorySamplingFailure:
            pass

        # Failure — return a no-op; next call will retry.
        return np.zeros(5, dtype=np.float32)

    def _do_push(self, state):
        """Try to execute the push skill."""
        obj_map = {o.name: o for o in state}
        robot = obj_map["robot"]
        hook = obj_map["hook"]
        movable = obj_map["movable_button"]
        target = obj_map["target_button"]

        if self._push_attempts >= self._max_skill_attempts:
            self._phase = "done"
            return np.zeros(5, dtype=np.float32)

        assert isinstance(self._action_space, CRVRobotActionSpace)
        controller = self._push_cls(
            objects=[robot, hook, movable, target],
            action_space=self._action_space,
            init_constant_state=self._init_constant_state,
        )
        params = controller.sample_parameters(state, self._rng)
        self._push_attempts += 1

        try:
            controller.reset(state, params)
            actions = []
            while not controller.terminated():
                actions.append(controller.step())
            self._action_queue = actions
            if self._action_queue:
                return self._action_queue.pop(0)
        except TrajectorySamplingFailure:
            pass

        # Failure — return a no-op; next call will retry.
        return np.zeros(5, dtype=np.float32)
