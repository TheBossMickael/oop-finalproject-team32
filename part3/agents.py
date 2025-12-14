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
            The current observation returned by env.reset() or env.step().

        Returns
        -------
        int
            The action to take (compatible with the environment's action space).
        """
        raise NotImplementedError("select_action() must be implemented by subclasses.")

    def reset(self) -> None:
        """
        Optional hook for agents that need to reset internal state
        at the beginning of a new episode.

        RandomAgent and GreedyAgent do not need it, so the default
        implementation does nothing.
        """
        pass


class RandomAgent(BaseAgent):
    """
    Agent that selects actions uniformly at random.

    This is a simple baseline agent. It ignores the observation and only
    uses the action_space to sample actions.
    """

    def select_action(self, observation) -> int:
        """
        Select a random action, ignoring the observation.
        """
        return self.action_space.sample()


class GreedyAgent(BaseAgent):
    """
    Simple heuristic agent for the WarehouseRobot environment.

    Assumes the observation is a 1D array or list with at least 4 values:
        [robot_row, robot_col, target_row, target_col, ...]

    The agent always tries to move closer to the target:
    - If the robot is above the target  -> move DOWN
    - If the robot is below the target  -> move UP
    - If on the same row and left of target  -> move RIGHT
    - If on the same row and right of target -> move LEFT

    Action indices follow the RobotAction enum in warehouse_robot.py:
        LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3
    """

    def select_action(self, observation) -> int:
        """
        Select an action that greedily moves the robot closer to the target.
        """
        # Unpack the first four elements of the observation
        robot_row, robot_col, target_row, target_col = observation[:4]

        candidate_actions = []

        # Vertical movement: align rows first
        if robot_row < target_row:
            # Robot is above the target -> move DOWN
            candidate_actions.append(1)
        elif robot_row > target_row:
            # Robot is below the target -> move UP
            candidate_actions.append(3)

        # Horizontal movement: align columns if on the same row
        if robot_row == target_row:
            if robot_col < target_col:
                # Robot is left of the target -> move RIGHT
                candidate_actions.append(2)
            elif robot_col > target_col:
                # Robot is right of the target -> move LEFT
                candidate_actions.append(0)

        # If no clear greedy action exists, fall back to a random action
        if not candidate_actions:
            return self.action_space.sample()

        # Randomly choose among equally good candidate actions
        return random.choice(candidate_actions)
