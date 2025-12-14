"""
Main script for Part 3: Warehouse Robot OOP project.

This script demonstrates:
- How to create a Gymnasium environment (basic or advanced)
- How to plug different agents (RandomAgent, GreedyAgent) into the same environment
- How to use the Trainer class to run training and evaluation loops
- How to switch environments without changing the Trainer or the agents
"""

from __future__ import annotations

from oop_project_env import WarehouseRobotEnv, AdvancedWarehouseRobotEnv
from agents import RandomAgent, GreedyAgent
from trainer import Trainer


# You can change these values if you want a different grid size.
DEFAULT_GRID_ROWS = 4
DEFAULT_GRID_COLS = 5
MAX_STEPS_PER_EPISODE = 50


def run_with_agent(
    agent_class,
    agent_name: str,
    env_class,
    env_name: str,
    num_train_episodes: int = 50,
    num_eval_episodes: int = 20,
) -> None:
    """
    Run training and evaluation with a given agent class on the selected environment.

    This function prints numerical results only (success rate, average reward, average steps),
    with no graphical rendering.
    """
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Agent: {agent_name}")
    print("=" * 60)

    # -------- Training phase (no rendering) --------
    env = env_class(
        grid_rows=DEFAULT_GRID_ROWS,
        grid_cols=DEFAULT_GRID_COLS,
        render_mode=None,
    )
    agent = agent_class(env.action_space)
    trainer = Trainer(env, agent, max_steps_per_episode=MAX_STEPS_PER_EPISODE)

    train_stats = trainer.train(num_episodes=num_train_episodes, render=False)

    # -------- Evaluation phase (no rendering) --------
    eval_env = env_class(
        grid_rows=DEFAULT_GRID_ROWS,
        grid_cols=DEFAULT_GRID_COLS,
        render_mode=None,
    )
    eval_agent = agent_class(eval_env.action_space)
    eval_trainer = Trainer(
        eval_env,
        eval_agent,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
    )

    eval_stats = eval_trainer.evaluate(num_episodes=num_eval_episodes, render=False)

    print()
    print(f"[{env_name} | {agent_name}] Train success rate: {train_stats['success_rate'] * 100:.2f}%")
    print(f"[{env_name} | {agent_name}] Train avg reward:    {train_stats['avg_reward']:.3f}")
    print(f"[{env_name} | {agent_name}] Train avg steps:     {train_stats['avg_steps']:.2f}")
    print()
    print(f"[{env_name} | {agent_name}] Eval success rate:  {eval_stats['success_rate'] * 100:.2f}%")
    print(f"[{env_name} | {agent_name}] Eval avg reward:    {eval_stats['avg_reward']:.3f}")
    print(f"[{env_name} | {agent_name}] Eval avg steps:     {eval_stats['avg_steps']:.2f}")
    print()


def choose_environment():
    """
    Interactive menu to choose which environment (game mode) to run.
    """
    print("Select environment:")
    print("  1) Basic Warehouse")
    print("  2) Advanced Warehouse (obstacles + battery)")
    choice = input("Your choice [1/2] (default=2): ").strip()

    if choice == "1":
        return WarehouseRobotEnv, "Basic"
    # Default
    return AdvancedWarehouseRobotEnv, "Advanced"


def run_visual_demo(
    agent_class,
    agent_name: str,
    env_class,
    env_name: str,
    num_episodes: int = 3,
) -> None:
    """
    Run a short visual demo with rendering enabled.

    This uses render_mode="human" so the environment opens the Pygame window
    and updates the grid at each step.
    """
    print("=" * 60)
    print("Visual demo")
    print(f"Environment: {env_name}")
    print(f"Agent: {agent_name}")
    print("Close the game window or press ESC to quit.")
    print("=" * 60)

    env = env_class(
        grid_rows=DEFAULT_GRID_ROWS,
        grid_cols=DEFAULT_GRID_COLS,
        render_mode="human",
    )
    agent = agent_class(env.action_space)
    trainer = Trainer(env, agent, max_steps_per_episode=MAX_STEPS_PER_EPISODE)

    trainer.evaluate(num_episodes=num_episodes, render=True)


def choose_agent():
    """
    Interactive menu to choose which agent to run.
    """
    print("Select agent:")
    print("  1) RandomAgent")
    print("  2) GreedyAgent")
    choice = input("Your choice [1/2] (default=2): ").strip()

    if choice == "1":
        return RandomAgent, "RandomAgent"
    # Default
    return GreedyAgent, "GreedyAgent"


def choose_run_mode():
    """
    Interactive menu to choose how to run the main script.
    """
    print()
    print("Select run mode:")
    print("  1) Statistics only (training + evaluation, no graphics)")
    print("  2) Visual demo only")
    print("  3) Both statistics and visual demo")
    choice = input("Your choice [1/2/3] (default=3): ").strip()

    if choice == "1":
        return "stats"
    elif choice == "2":
        return "visual"
    # Default
    return "both"


if __name__ == "__main__":
    env_class, env_name = choose_environment()
    agent_class, agent_name = choose_agent()
    run_mode = choose_run_mode()

    if run_mode in ("stats", "both"):
        run_with_agent(agent_class, agent_name, env_class=env_class, env_name=env_name)

    if run_mode in ("visual", "both"):
        run_visual_demo(agent_class, agent_name, env_class=env_class, env_name=env_name, num_episodes=3)
