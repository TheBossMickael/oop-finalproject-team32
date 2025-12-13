"""
Agents module for the Warehouse Robot project (Part 3).

This file defines a small hierarchy of agents that can act in any
Gymnasium environment with a discrete action space.

OOP goals:
- BaseAgent: abstract base class (encapsulation + inheritance)
- RandomAgent: simple baseline agent (inherits from BaseAgent)
- GreedyAgent: heuristic agent that uses the observation to decide
  which action to take (polymorphism: same interface, different behavior).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import random


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Each agent must implement the select_action() method.
    The agent only knows the action_space and receives observations
    from the environment. It does not know any environment internals.
    """

    def __init__(self, action_space: Any) -> None:
        """
        Initialize the agent.

        Parameters
        ----------
        action_space : gymnasium.Space
            The action space of the environment. We keep it generic (Any)
            so that BaseAgent can work with any Gym-compatible environment.
        """
        self.action_space = action_space

    @abstractmethod
    def select_action(self, observation) -> int:
        """
        Select an action given the current observation.

        This method must be overridden by all subclasses.

        Parameters
        ----------
        observation :
            The current state/observation returned by env.reset() or env.step().

        Returns
        -------
        int
            The action to take (compatible with the env's action space).
        """
        raise NotImplementedError("select_action() must be implemented by subclasses.")

    def reset(self) -> None:
        """
        Optional hook for agents that need to reset their internal state
        at the beginning of a new episode.

        RandomAgent and GreedyAgent do not need it, so the default
        implementation does nothing. Subclasses can override it.
        """
        pass


class RandomAgent(BaseAgent):
    """
    Agent that selects actions uniformly at random.

    This is a simple baseline agent. It ignores the observation and only
    uses the action_space to sample actions.
    """

    def select_action(self, observation) -> int:
        # Completely ignore the observation and sample a random action.
        return self.action_space.sample()


class GreedyAgent(BaseAgent):
    """
    Simple heuristic agent for the WarehouseRobot environment.

    Assumes the observation is a 1D array or list of length 4:
        [robot_row, robot_col, target_row, target_col]

    The agent always tries to move closer to the target:
    - If robot is above the target  -> move DOWN
    - If robot is below the target  -> move UP
    - Else if robot is left of target  -> move RIGHT
    - Else if robot is right of target -> move LEFT

    Action indices follow the RobotAction enum in warehouse_robot.py:
        LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3
    """

    def select_action(self, observation) -> int:
        # Unpack observation. We accept either list, tuple, or numpy array.
        robot_row, robot_col, target_row, target_col = observation

        candidate_actions = []

        # Vertical preference: first try to align rows
        if robot_row < target_row:
            # Robot is above the target -> go DOWN (1)
            candidate_actions.append(1)
        elif robot_row > target_row:
            # Robot is below the target -> go UP (3)
            candidate_actions.append(3)

        # Horizontal adjustment: if on the same row, adjust columns
        if robot_row == target_row:
            if robot_col < target_col:
                # Robot is left of target -> go RIGHT (2)
                candidate_actions.append(2)
            elif robot_col > target_col:
                # Robot is right of target -> go LEFT (0)
                candidate_actions.append(0)

        # If the robot is already exactly on the target, or if for some reason
        # no candidate action was selected, fall back to a random action.
        if not candidate_actions:
            return self.action_space.sample()

        # If there are multiple good candidate actions, pick one at random
        # among them to avoid always following the exact same path.
        return random.choice(candidate_actions)
