'''
Custom Gym environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import warehouse_robot as wr
import numpy as np


# Register the basic environment
register(
    id='warehouse-robot-v0',
    entry_point='oop_project_env:WarehouseRobotEnv',
)

# Register the advanced environment
register(
    id='warehouse-robot-advanced-v1',
    entry_point='oop_project_env:AdvancedWarehouseRobotEnv',
)



# Basic Warehouse Robot Environment
class WarehouseRobotEnv(gym.Env):
    """
    Simple Gym environment for a warehouse robot.

    The robot moves on a grid and must reach a target.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_rows=4, grid_cols=5, render_mode=None):

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode

        # Initialize the low-level warehouse robot (grid + rendering)
        self.warehouse_robot = wr.WarehouseRobot(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            fps=self.metadata["render_fps"]
        )

        # Action space: LEFT, DOWN, RIGHT, UP
        self.action_space = spaces.Discrete(len(wr.RobotAction))

        # Observation space:
        # [robot_row, robot_col, target_row, target_col]
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([
                self.grid_rows - 1,
                self.grid_cols - 1,
                self.grid_rows - 1,
                self.grid_cols - 1
            ]),
            shape=(4,),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation.
        """
        super().reset(seed=seed)

        # Reset the robot and randomly place the target
        self.warehouse_robot.reset(seed=seed)

        # Build observation
        obs = np.concatenate(
            (self.warehouse_robot.robot_pos, self.warehouse_robot.target_pos)
        )

        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        """
        Apply an action and return the environment transition.
        """
        # Apply action to the robot
        target_reached = self.warehouse_robot.perform_action(
            wr.RobotAction(action)
        )

        reward = 0
        terminated = False

        # Success condition
        if target_reached:
            reward = 1
            terminated = True

        # Build next observation
        obs = np.concatenate(
            (self.warehouse_robot.robot_pos, self.warehouse_robot.target_pos)
        )

        info = {}

        if self.render_mode == "human":
            print(wr.RobotAction(action))
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        """
        Render the environment using the warehouse robot renderer.
        """
        self.warehouse_robot.render()


# Advanced Warehouse Robot Environment
class AdvancedWarehouseRobotEnv(WarehouseRobotEnv):
    """
    Advanced version of the warehouse environment.

    Adds:
    - Obstacles
    - Battery constraint
    """

    def __init__(
        self,
        grid_rows=4,
        grid_cols=5,
        render_mode=None,
        max_battery=30,
        num_obstacles=4
    ):
        super().__init__(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            render_mode=render_mode
        )

        self.max_battery = max_battery
        self.num_obstacles = num_obstacles
        self.battery = max_battery
        self.obstacles = []

        # Observation space:
        # [robot_row, robot_col, target_row, target_col, battery]
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([
                self.grid_rows - 1,
                self.grid_cols - 1,
                self.grid_rows - 1,
                self.grid_cols - 1,
                self.max_battery
            ]),
            shape=(5,),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment, battery level, and obstacles.
        """
        super().reset(seed=seed)
        self.battery = self.max_battery

        rng = np.random.default_rng(seed)
        self.obstacles = []

        # Randomly place obstacles (not on robot or target)
        while len(self.obstacles) < self.num_obstacles:
            r = int(rng.integers(0, self.grid_rows))
            c = int(rng.integers(0, self.grid_cols))
            pos = (r, c)

            if list(pos) == self.warehouse_robot.robot_pos:
                continue
            if list(pos) == self.warehouse_robot.target_pos:
                continue
            if pos in self.obstacles:
                continue

            self.obstacles.append(pos)

        # Send advanced info to the renderer 
        self.warehouse_robot.obstacles = self.obstacles
        self.warehouse_robot.battery = self.battery

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        """
        Apply an action with obstacle and battery constraints.
        """
        old_pos = tuple(self.warehouse_robot.robot_pos)

        target_reached = self.warehouse_robot.perform_action(
            wr.RobotAction(action)
        )
        new_pos = tuple(self.warehouse_robot.robot_pos)

        reward = 0
        terminated = False

        # Obstacle collision: cancel movement and apply penalty
        if new_pos in self.obstacles:
            self.warehouse_robot.robot_pos = list(old_pos)
            reward = -0.2

        # Battery consumption
        self.battery -= 1
        if self.battery <= 0:
            reward = -1
            terminated = True

        # Success condition
        if target_reached:
            reward = 1
            terminated = True

        # Update renderer info every step
        self.warehouse_robot.obstacles = self.obstacles
        self.warehouse_robot.battery = self.battery

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def _get_obs(self):
        """
        Build the observation vector.
        """
        return np.array([
            self.warehouse_robot.robot_pos[0],
            self.warehouse_robot.robot_pos[1],
            self.warehouse_robot.target_pos[0],
            self.warehouse_robot.target_pos[1],
            self.battery
        ], dtype=np.int32)


# --------------------------------------------------
# Unit test
# --------------------------------------------------
if __name__ == "__main__":    
    
    env = gym.make("warehouse-robot-v0", render_mode="human")

    obs = env.reset()[0]

    while True:
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        if terminated:
            obs = env.reset()[0]
