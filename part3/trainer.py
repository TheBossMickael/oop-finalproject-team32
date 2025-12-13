"""
Trainer module for the Warehouse Robot project (Part 3).

The Trainer class is responsible for running episodes in a Gymnasium
environment using any Agent that inherits from BaseAgent.

OOP goals:
- Encapsulation of the training / evaluation loop logic inside a Trainer class.
- Polymorphism: the Trainer works with any environment that follows the
  Gymnasium API (reset, step, render) and any Agent that implements
  select_action().
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from agents import BaseAgent


class Trainer:
    """
    Orchestrates interactions between an environment and an agent.

    The Trainer does not know the internal details of the environment
    (grid, sprites, rewards) or the agent (random, greedy, etc.).
    It only relies on:
      - env.reset(), env.step(action), env.render()  (Gym API)
      - agent.select_action(observation), agent.reset() (BaseAgent API)

    This demonstrates polymorphism: as long as env and agent respect
    these interfaces, the Trainer can run episodes with them.
    """

    def __init__(
        self,
        env: Any,
        agent: BaseAgent,
        max_steps_per_episode: int | None = None,
    ) -> None:
        """
        Initialize the Trainer.

        Parameters
        ----------
        env : Gymnasium-like environment
            Any environment that provides reset(), step(action) and optionally render().
        agent : BaseAgent
            Any agent that implements select_action(observation).
        max_steps_per_episode : int or None
            Optional safety limit on the number of steps per episode.
            If None, only the environment's terminated/truncated flags stop the episode.
        """
        self.env = env
        self.agent = agent
        self.max_steps_per_episode = max_steps_per_episode

    def run_episode(
        self,
        train: bool = False,
        render: bool = False,
    ) -> Tuple[float, bool, int]:
        """
        Run a single episode.

        Parameters
        ----------
        train : bool
            Kept for future extension (e.g., Q-learning). For RandomAgent and GreedyAgent,
            there is no learning, but we keep the flag to show where training would happen.
        render : bool
            If True, calls env.render() at each step (use with a 'human' render_mode env).

        Returns
        -------
        total_reward : float
            Sum of rewards obtained during the episode.
        success : bool
            True if the episode is considered successful (e.g., reached the goal).
        steps : int
            Number of steps taken in this episode.
        """
        # Reset environment and agent
        obs, info = self.env.reset()
        self.agent.reset()

        done = False
        total_reward = 0.0
        steps = 0
        success = False

        while not done:
            # Get action from agent (polymorphic call)
            action = self.agent.select_action(obs)

            # Apply action to environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward
            steps += 1

            # Optional visualization
            if render and hasattr(self.env, "render"):
                self.env.render()

            # Define episode termination
            done = terminated or truncated

            # Define success: here, any positive reward ends the episode with success
            if terminated and reward > 0:
                success = True

            # Optional safety limit on steps per episode
            if self.max_steps_per_episode is not None and steps >= self.max_steps_per_episode:
                done = True

        return total_reward, success, steps

    def train(
        self,
        num_episodes: int,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Run multiple episodes in 'training' mode.

        For RandomAgent and GreedyAgent there is no learning, but this method
        is still useful to collect statistics and compare different agents
        or environments.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to run.
        render : bool
            If True, render each step (not recommended for large num_episodes).

        Returns
        -------
        stats : dict
            Dictionary with aggregated statistics:
            - 'num_episodes'
            - 'success_rate'
            - 'avg_reward'
            - 'avg_steps'
        """
        total_rewards = 0.0
        total_steps = 0
        num_success = 0

        for episode in range(num_episodes):
            total_reward, success, steps = self.run_episode(train=True, render=render)

            total_rewards += total_reward
            total_steps += steps
            if success:
                num_success += 1

        success_rate = num_success / num_episodes if num_episodes > 0 else 0.0
        avg_reward = total_rewards / num_episodes if num_episodes > 0 else 0.0
        avg_steps = total_steps / num_episodes if num_episodes > 0 else 0.0

        stats = {
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
        }

        # Basic console summary (useful during the demo)
        print(f"[TRAIN] Episodes: {num_episodes}")
        print(f"[TRAIN] Success rate: {success_rate * 100:.2f}%")
        print(f"[TRAIN] Average reward: {avg_reward:.3f}")
        print(f"[TRAIN] Average steps: {avg_steps:.2f}")

        return stats

    def evaluate(
        self,
        num_episodes: int,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Run multiple episodes in 'evaluation' mode.

        This is similar to train(), but conceptually used when the agent
        is not supposed to learn anymore. For RandomAgent and GreedyAgent,
        there is no difference in behavior, but the separation is useful
        for future learning agents.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to run.
        render : bool
            If True, render each step.

        Returns
        -------
        stats : dict
            Same fields as train():
            - 'num_episodes'
            - 'success_rate'
            - 'avg_reward'
            - 'avg_steps'
        """
        total_rewards = 0.0
        total_steps = 0
        num_success = 0

        for episode in range(num_episodes):
            total_reward, success, steps = self.run_episode(train=False, render=render)

            total_rewards += total_reward
            total_steps += steps
            if success:
                num_success += 1

        success_rate = num_success / num_episodes if num_episodes > 0 else 0.0
        avg_reward = total_rewards / num_episodes if num_episodes > 0 else 0.0
        avg_steps = total_steps / num_episodes if num_episodes > 0 else 0.0

        stats = {
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
        }

        print(f"[EVAL] Episodes: {num_episodes}")
        print(f"[EVAL] Success rate: {success_rate * 100:.2f}%")
        print(f"[EVAL] Average reward: {avg_reward:.3f}")
        print(f"[EVAL] Average steps: {avg_steps:.2f}")

        return stats
