import random
from typing import Dict, Any, Tuple

from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import BUTTON_MIDDLE

from safe_transfer.environments.drone_2d_budget import Drone2dBudgetEnv
from safe_transfer.environments.utils import clip_norm, distance_normalized


class Drone2dBudgetPackageEnv(Drone2dBudgetEnv):
    """
    Drone Environment with Battery and Package Delivery Task.

    Inherits from Drone2dBudgetEnv.

    Task: Go to package location, "collect" it, then go to target location.

    Adds:
        - Package position.
        - Package collected state.
        - Package info in observation space.
        - Modified reward function for two-phase task.
        - Rendering for package location/state.
        - Event handling for changing package position.

    Observation Space (normalized):
        - [Base Observations from Drone2dBudgetEnv]
        - package_pos_x
        - package_pos_y
        - package_collected_flag (0 or 1)

    Args:
        package_collect_threshold (float): Normalized distance to consider package collected.
        change_package (bool): Allow changing package position with middle mouse click.
        reward_package (float): Bonus reward for collecting the package.
        (Inherits other args from Drone2dBudgetEnv)
    """
    def __init__(self,
                 render_mode: str | None = None,
                 package_collect_threshold=0.10,
                 change_package=True,
                 reward_package=30.0,  # Bonus reward for collecting package
                 **kwargs): # Pass args to parent class

        super().__init__(render_mode=render_mode, **kwargs)

        self.package_collect_threshold = package_collect_threshold
        self.change_package = change_package
        self.reward_package = reward_package

        # Package state
        self.package_collected = False
        self.x_package = 0.0
        self.y_package = 0.0

        # Finalize observation space (extend again)
        self._init_observation_space() # Overrides parent method call

        # Update info keys again
        self.info_keys = list(self.info_keys) + ["package_collected"] # Add package specific info key

    # --- Override Observation Space ---
    def _init_observation_space(self):
        """Initializes the extended observation space including package info."""
        parent_obs_len = 11
        num_obs = parent_obs_len + 3 # Add package_x, package_y, collected_flag
        min_observation = np.array([-1] * num_obs, dtype=np.float32)
        max_observation = np.array([1] * num_obs, dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)

    # --- Override Observation Getter ---
    def get_observation(self) -> np.ndarray:
        """Computes the observation vector including package info."""
        budget_obs = super().get_observation() # Gets obs from Drone2dBudgetEnv

        # Normalize package position
        norm_package_x = clip_norm(self.x_package, self.window_size)
        norm_package_y = clip_norm(self.y_package, self.window_size)

        # Package collected flag (normalized to {0, 1})
        collected_flag = 1.0 if self.package_collected else 0.0

        return np.concatenate([
            budget_obs,
            np.array([norm_package_x, norm_package_y, collected_flag], dtype=np.float32)
        ])

    # --- Override Reward Calculation ---
    def _calculate_reward(self, obs: np.ndarray) -> float:
        """
        Calculates reward based on the current phase (collect package or deliver).
        Always prioritizes battery if low.
        """
        # Unpack observation (all 14 elements)
        norm_pos_x, norm_pos_y = obs[4:6]
        norm_target_x, norm_target_y = obs[6:8]
        norm_battery_x, norm_battery_y, battery_ratio = obs[8:11]
        norm_package_x, norm_package_y, collected_flag = obs[11:14]

        dist_target = distance_normalized(norm_pos_x, norm_pos_y, norm_target_x, norm_target_y)
        dist_battery = distance_normalized(norm_pos_x, norm_pos_y, norm_battery_x, norm_battery_y)
        dist_package = distance_normalized(norm_pos_x, norm_pos_y, norm_package_x, norm_package_y)

        reward = 0.0

        # --- Phase 1: Collect Package ---
        if not self.package_collected: # collected_flag should be 0
            package_reached = dist_package <= self.package_collect_threshold

            if package_reached:
                self.package_collected = True
                self.info['package_collected'] += 1
                self.info['task_completed'] += 1
                 # Give bonus only if battery is sufficient for next phase (heuristic)
                if battery_ratio >= self.battery_low_threshold:
                     reward = self.reward_package # Intermediate bonus for collecting package
                else:
                    # Collected package but low battery, reward based on getting to battery
                    reward = np.exp(-self.reward_distance_beta * dist_battery)
            else:
                # Package not reached, prioritize based on battery
                if battery_ratio >= self.battery_low_threshold:
                    # Sufficient battery: reward based on distance to package
                    reward = np.exp(-self.reward_distance_beta * dist_package)
                else:
                    # Low battery: reward based on distance to battery station
                    reward = np.exp(-self.reward_distance_beta * dist_battery)

        # --- Phase 2: Deliver Package ---
        else: # collected_flag should be 1
            target_reached = dist_target <= self.target_reach_threshold

            if target_reached:
                # Package delivered! Task complete for this cycle.
                self.info['goals_reached'] += 1 # Reached final target
                self.info['task_completed'] += 1 # Increment task counter

                # Give bonus only if battery is sufficient
                if battery_ratio >= self.battery_low_threshold:
                    reward = self.reward_goal # Final delivery bonus
                else:
                    # Delivered but low battery, reward based on getting to battery
                    reward = np.exp(-self.reward_distance_beta * dist_battery)

                # Reset for next package cycle if allowed
                if self.change_target or self.change_package:
                    self._reset_task_state() # Resets package collected flag, generates new package/target

            else:
                # Target not reached, prioritize based on battery
                if battery_ratio >= self.battery_low_threshold:
                    # Sufficient battery: reward based on distance to target
                    reward = np.exp(-self.reward_distance_beta * dist_target)
                else:
                    # Low battery: reward based on distance to battery station
                    reward = np.exp(-self.reward_distance_beta * dist_battery)

        return float(reward)

    def _reset_task_state(self):
        """Resets package state and generates new positions for package and target."""
        self.package_collected = False
        self._generate_package()
        # Also generate a new target for the delivery phase
        self._generate_target()

    # --- Override Reset ---
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the full environment including package state."""
        # Call parent reset (handles base + battery)
        observation_budget, info_budget = super().reset(seed=seed, options=options)

        # Reset package specific state
        self._reset_task_state() # Generates package pos and sets collected=False

        # Re-initialize info dict with all keys for this env
        self.info = {k: 0 for k in self.info_keys}
        # Copy over any info potentially set by parent reset
        self.info.update(info_budget)

        # Get the full observation including new package state
        observation = self.get_observation()

        return observation, self.info

    def _generate_package(self):
        """Generates a random package position."""
        margin = 100
        # Ensure package doesn't spawn too close to target initially
        while True:
            self.x_package = random.uniform(margin, self.window_size - margin)
            self.y_package = random.uniform(margin, self.window_size - margin)
            dist_sq = (self.x_package - self.x_target)**2 + (self.y_package - self.y_target)**2
            if dist_sq > (2 * margin)**2: # Ensure they are reasonably far apart
                break


    def _render_environment_elements(self):
        """Renders budget env elements (base + battery) + package."""
        # First, render everything the parent (BudgetEnv) renders in the environment
        super()._render_environment_elements()

        # Then, add the package rendering (if not collected)
        self._render_package()

    # --- Specific Rendering Method for Package Env ---

    def _render_package(self):
        """Renders the package location."""
        if not self.screen or self.package_collected: return # Don't draw if collected

        package_screen_x = int(self.x_package)
        package_screen_y = int(self.window_size - self.y_package) # Convert y
        # Calculate radius based on normalized threshold
        radius_pixels = int(self.package_collect_threshold * (self.window_size / 2))

        surface = pygame.Surface((radius_pixels * 2, radius_pixels * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (0, 0, 255, 100), (radius_pixels, radius_pixels), radius_pixels) # Blue, semi-transparent
        self.screen.blit(surface, (package_screen_x - radius_pixels, package_screen_y - radius_pixels))


    # Add the event hook override (if not already present from previous refactoring)
    def _handle_other_mouse_buttons(self, button, world_x, world_y):
        """Handles middle-click for package placement."""
        if button == BUTTON_MIDDLE:
            self.change_package_point(world_x, world_y)
        else:
            # If there were other buttons handled by the parent, call super:
            # super()._handle_other_mouse_buttons(button, world_x, world_y)
            pass # In this case, BudgetEnv doesn't use this hook itself.

    # --- Point Manipulation Helper ---
    def change_package_point(self, world_x: float, world_y: float):
        """Sets the package position."""
        self.x_package = world_x
        self.y_package = world_y
        self.package_collected = False # Reset collected status if moved manually
        print(f"Package moved to: ({world_x:.1f}, {world_y:.1f})")


if __name__ == "__main__":
    print("\nTesting Drone2dBudgetPackageEnv...")
    env_package = Drone2dBudgetPackageEnv(render_mode='human', n_steps=2000,
                                           change_target=True, change_battery_station=True, change_package=True)
    obs, info = env_package.reset(seed=44)
    terminated = truncated = False
    total_reward_package = 0

    for i in range(env_package.max_time_steps):
        action = env_package.action_space.sample() # Replace with your agent's action selection
        obs, reward, terminated, truncated, info = env_package.step(action)
        total_reward_package += reward

        if terminated or truncated:
            print(f"Episode {i+1} finished with reward: {total_reward_package:.2f}, Info: {info}")
            # Optional: Reset for another episode test
            # obs, info = env_package.reset(seed=random.randint(0, 1000))
            # total_reward_package = 0
            # terminated = truncated = False
            break # Stop after one episode for this test script
