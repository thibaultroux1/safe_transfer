import sys
import random
from typing import Dict, Any, Tuple

from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, MOUSEBUTTONUP, BUTTON_LEFT, BUTTON_RIGHT)

from safe_transfer.environments.drone_2d import Drone2dEnv
from safe_transfer.environments.utils import clip_norm, distance_normalized


class Drone2dBudgetEnv(Drone2dEnv):
    """
    Drone Environment with a limited battery resource.

    Inherits from Drone2dEnv.

    Adds:
        - Battery state (current, max, cooldown).
        - Battery station position.
        - Battery info in observation space.
        - Modified reward function (prioritizes battery when low).
        - Termination condition for running out of battery.
        - Rendering for battery gauge, cooldown, and station.
        - Event handling for changing battery station position.

    Observation Space (normalized):
        - [Base Observations from Drone2dEnv]
        - battery_station_pos_x
        - battery_station_pos_y
        - battery_level_ratio [0, 1]

    Args:
        initial_battery (int): Starting battery level (in steps).
        max_battery (int): Maximum battery level.
        battery_cooldown_duration (int): Steps cooldown after refilling battery.
        battery_low_threshold (float): Normalized battery ratio below which reward prioritizes battery station.
        battery_station_reach_threshold (float): Normalized distance to consider battery station reached.
        change_battery_station (bool): Allow changing battery station with right mouse click.
        penalty_ruin (float): Penalty for running out of battery.
        (Inherits other args from Drone2dEnv)
    """
    def __init__(self,
                 render_mode: str | None = None,
                 initial_battery=500,
                 max_battery=500,
                 battery_cooldown_duration=200,
                 battery_low_threshold=0.5,
                 battery_station_reach_threshold=0.10,
                 change_battery_station=True,
                 penalty_ruin=-100.0,  # Penalty for running out of battery
                 **kwargs): # Pass other args to base class

        # Call base class constructor first
        super().__init__(render_mode=render_mode, **kwargs)

        # Battery specific parameters
        self.initial_battery = initial_battery
        self.max_battery = max_battery
        self.battery_cooldown_duration = battery_cooldown_duration
        self.battery_low_threshold = battery_low_threshold
        self.battery_station_reach_threshold = battery_station_reach_threshold
        self.change_battery_on_click = change_battery_station
        self.penalty_ruin = penalty_ruin  # Store the ruin penalty

        # Battery state variables (reset properly)
        self.battery = self.initial_battery
        self.battery_cooldown = 0
        self._in_battery_zone = False # Internal flag

        # Battery station position (raw coordinates)
        self.x_battery = 0.0
        self.y_battery = 0.0

        # Finalize observation space (extend base)
        self._init_observation_space() # Overrides base method call during super().__init__

        # Update info keys
        self.info_keys = list(self.info_keys) + ["ruin", "battery_refill"]

    # --- Override Observation Space ---
    def _init_observation_space(self):
        """Initializes the extended observation space including battery info."""
        base_obs_len = 8
        num_obs = base_obs_len + 3 # Add battery_x, battery_y, battery_ratio
        min_observation = np.array([-1] * num_obs, dtype=np.float32)
        max_observation = np.array([1] * num_obs, dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)

    # --- Override Observation Getter ---
    def get_observation(self) -> np.ndarray:
        """Computes the observation vector including battery info."""
        base_obs = super().get_observation()

        # Normalize battery station position
        norm_battery_x = clip_norm(self.x_battery, self.window_size)
        norm_battery_y = clip_norm(self.y_battery, self.window_size)

        # Normalize battery level
        battery_ratio = np.clip(self.battery / self.max_battery, 0.0, 1.0)

        return np.concatenate([
            base_obs,
            np.array([norm_battery_x, norm_battery_y, battery_ratio], dtype=np.float32)
        ])

    # --- Override Step Logic ---
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Runs one timestep, including battery updates and modified reward/termination."""
        # Core physics step from base class
        self._apply_action_and_step_physics(action)
        self._update_paths_and_shades()

        if self._first_step: self._first_step = False

        obs = self.get_observation()

        # Update battery state *before* calculating reward/termination
        self._update_battery(obs)

        # Calculate reward using overridden method
        reward = self._calculate_reward(obs)

        # Check termination using overridden method (includes battery check)
        terminated, truncated = self._check_termination(obs)

        # Apply penalty for termination (crash or ruin)
        if terminated:
            if self.info['ruin'] == 1:
                reward = self.penalty_ruin
            elif self.info['crash'] == 1:
                reward = self.penalty_crash

        self.reward_value = reward # Store for rendering

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self.info


    def _update_battery(self, obs: np.ndarray):
        """Updates battery level, checks for refill, and handles cooldown."""
        # Unpack necessary components (positions are normalized)
        norm_pos_x, norm_pos_y = obs[4:6]
        norm_battery_x, norm_battery_y = obs[8:10]

        dist_to_battery = distance_normalized(norm_pos_x, norm_pos_y, norm_battery_x, norm_battery_y)
        currently_in_zone = dist_to_battery <= self.battery_station_reach_threshold

        # Refill logic
        if currently_in_zone and self.battery < self.max_battery and self.battery_cooldown == 0:
            self.battery = self.max_battery
            self.info['battery_refill'] += 1
            self.battery_cooldown = self.battery_cooldown_duration # Start cooldown *after* refill
        else:
            # Deplete battery if not refilling (or if in cooldown/already full)
            self.battery -= 1

        # Cooldown logic (start cooldown when *leaving* the zone after being inside)
        # if self._in_battery_zone and not currently_in_zone and self.battery_cooldown == 0:
        #    self.battery_cooldown = self.battery_cooldown_duration
        # Update internal flag *after* checking refill/cooldown logic for the step
        self._in_battery_zone = currently_in_zone

        # Decrement cooldown timer
        if self.battery_cooldown > 0:
            self.battery_cooldown -= 1


    # --- Override Reward Calculation ---
    def _calculate_reward(self, obs: np.ndarray) -> float:
        """Calculates reward, prioritizing battery when low, target otherwise."""
        # Unpack observation
        norm_pos_x, norm_pos_y = obs[4:6]
        norm_target_x, norm_target_y = obs[6:8]
        norm_battery_x, norm_battery_y, battery_ratio = obs[8:11]

        dist_target = distance_normalized(norm_pos_x, norm_pos_y, norm_target_x, norm_target_y)
        dist_battery = distance_normalized(norm_pos_x, norm_pos_y, norm_battery_x, norm_battery_y)

        reward = 0.0

        # Check if target reached
        target_reached = dist_target <= self.target_reach_threshold
        if target_reached:
            # Generate new target only if allowed
            if self.change_target:
                self._generate_target() # Generate new target immediately
            self.info['goals_reached'] += 1
            self.info['task_completed'] += 1 # Count reaching target as task completion
            # Give bonus only if battery is sufficient
            if battery_ratio >= self.battery_low_threshold:
                reward = self.reward_goal # Large bonus
            else:
                # Reached target but low battery, reward is based on getting to battery
                reward = np.exp(-self.reward_distance_beta * dist_battery)
        else:
            # Target not reached, base reward depends on battery level
            if battery_ratio >= self.battery_low_threshold:
                # Sufficient battery: reward based on distance to target
                reward = np.exp(-self.reward_distance_beta * dist_target)
            else:
                # Low battery: reward based on distance to battery station
                reward = np.exp(-self.reward_distance_beta * dist_battery)

        return float(reward)


    # --- Override Termination Check ---
    def _check_termination(self, obs: np.ndarray) -> Tuple[bool, bool]:
        """Checks base termination conditions and adds battery depletion."""
        # Check base conditions (crash, flip, out of bounds) first
        terminated_base, truncated = super()._check_termination(obs)

        if terminated_base or truncated:
            return terminated_base, truncated # Already terminated or truncated

        # Check for battery depletion
        battery_depleted = self.battery <= 0
        if battery_depleted:
            self.info['ruin'] += 1 # Specific info key for battery death
            terminated = True
        else:
            terminated = False

        return terminated, truncated


    # --- Override Reset ---
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets base environment and battery-specific state."""
        # Call base reset first (handles pymunk, paths, base info keys, etc.)
        observation_base, info_base = super().reset(seed=seed, options=options)

        # Reset battery state
        self.battery = self.initial_battery
        self.battery_cooldown = 0
        self._in_battery_zone = False

        # Generate initial battery station position
        self._generate_battery()

        # Re-initialize info dict with all keys for this env
        self.info = {k: 0 for k in self.info_keys}
        # Copy over any info potentially set by base reset (like initial crash if throw is extreme)
        self.info.update(info_base)


        # Get the full observation including new battery state
        observation = self.get_observation()

        return observation, self.info

    def _generate_battery(self):
        """Generates a random battery station position."""
        margin = 100
        self.x_battery = random.uniform(margin, self.window_size - margin)
        self.y_battery = random.uniform(margin, self.window_size - margin)


    def _render_environment_elements(self):
        """Renders base elements + battery station."""
        # First, render everything the parent class renders in the environment
        super()._render_environment_elements()

        # Then, add the battery station rendering
        self._render_battery_station()

    def _render_hud_elements(self):
        """Renders the specific HUD for the Budget Env (gauge, cooldown, text)."""
        # NOTE: We DO NOT call super()._render_hud_elements() here because
        # we are completely replacing the HUD layout.

        if self.screen is None or self.font is None:
            return

        # Draw the battery gauge
        self._render_battery_gauge()

        # Draw the cooldown indicator (if active)
        self._render_cooldown_indicator()

        # Draw the text info using the layout specific to this class
        # self._render_text_info() # Calls the overridden _render_text_info below

    # --- Specific Rendering Methods for Budget Env ---

    def _render_battery_station(self):
        """Renders the battery station area."""
        if not self.screen: return
        station_screen_x = int(self.x_battery)
        station_screen_y = int(self.window_size - self.y_battery) # Convert y
        # Calculate radius based on normalized threshold
        radius_pixels = int(self.battery_station_reach_threshold * (self.window_size / 2))

        surface = pygame.Surface((radius_pixels * 2, radius_pixels * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (255, 165, 0, 100), (radius_pixels, radius_pixels), radius_pixels) # Orange, semi-transparent
        self.screen.blit(surface, (station_screen_x - radius_pixels, station_screen_y - radius_pixels))

    def _render_battery_gauge(self):
        """Draws the battery level gauge."""
        if not self.screen or not self.font: return

        battery_ratio = np.clip(self.battery / self.max_battery, 0.0, 1.0)
        gauge_x = 10
        gauge_y = 10 # Position at top-left
        gauge_width = 200
        gauge_height = 40

        if battery_ratio >= 0.7: color = (0, 255, 0) # Green
        elif battery_ratio >= 0.3: color = (255, 255, 0) # Yellow
        else: color = (255, 0, 0) # Red

        pygame.draw.rect(self.screen, (0, 0, 0), (gauge_x, gauge_y, gauge_width, gauge_height), 1)
        filled_width = int((gauge_width - 2) * battery_ratio)
        if filled_width > 0:
             pygame.draw.rect(self.screen, color, (gauge_x + 1, gauge_y + 1, filled_width, gauge_height - 2))


    def _render_cooldown_indicator(self):
        """Draws an indicator when battery refill is on cooldown."""
        if not self.screen or not self.font or self.battery_cooldown <= 0: return

        indicator_x = 220 # Position next to battery gauge
        indicator_y = 20
        radius = 10

        pygame.draw.circle(self.screen, (255, 140, 0), (indicator_x + radius, indicator_y + radius), radius) # Orange circle


    # --- Override Text Info Rendering for specific layout ---
    def _render_text_info(self):
        """Renders text info, shifted down for battery gauge."""
        if not self.font or not self.drone: return

        y_offset = 30 # Start text lower to make space for gauge/cooldown

        reward_text = self.font.render(f'Reward: {self.reward_value:.2f}', True, (0, 0, 0))
        self.screen.blit(reward_text, (10, 10 + y_offset))

        obs = self.get_observation()
        # Check obs length for safety
        if len(obs) >= 11:
            norm_pos_x, norm_pos_y = obs[4:6]
            norm_target_x, norm_target_y = obs[6:8]
            norm_battery_x, norm_battery_y = obs[8:10]

            dist_target = distance_normalized(norm_pos_x, norm_pos_y, norm_target_x, norm_target_y)
            dist_battery = distance_normalized(norm_pos_x, norm_pos_y, norm_battery_x, norm_battery_y)

            distance_text = self.font.render(f'D_Tgt(N): {dist_target:.2f} D_Bat(N): {dist_battery:.2f}', True, (0, 0, 0))
            self.screen.blit(distance_text, (10, 40 + y_offset))

            position_text = self.font.render(f'Pos(N): ({norm_pos_x:.2f}, {norm_pos_y:.2f})', True, (0, 0, 0))
            self.screen.blit(position_text, (10, 70 + y_offset))
        else:
             err_text = self.font.render('Obs N/A', True, (255, 0, 0))
             self.screen.blit(err_text, (10, 40 + y_offset))

    # Note: The main `render` method itself doesn't need to be overridden now,
    # as the base class `render` will call these overridden helpers automatically.
    # Keep the overridden _handle_pygame_events for the right-click functionality.

    # --- Override Event Handling ---
    def _handle_pygame_events(self):
        """Handles base events plus right-click for battery station."""
        if not self.render_mode == "human": return

        for event in pygame.event.get():
            if event.type == QUIT:
                self.close()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self.close()
                sys.exit()
            elif event.type == MOUSEBUTTONUP:
                 x, y = event.pos
                 world_y = self.window_size - y # Convert y coord once

                 if event.button == BUTTON_LEFT:
                     self.change_target_point(x, world_y)
                 elif event.button == BUTTON_RIGHT:
                     self.change_battery_point(x, world_y)
                 # Allow subclasses to handle other buttons
                 else:
                     self._handle_other_mouse_buttons(event.button, x, world_y)

    def _handle_other_mouse_buttons(self, button, world_x, world_y):
        """Placeholder for subclasses to handle middle click etc."""
        pass # Drone2dBudgetPackageEnv will override this


    # --- Point Manipulation Helper ---
    def change_battery_point(self, world_x: float, world_y: float):
        """Sets the battery station position."""
        self.x_battery = world_x
        self.y_battery = world_y
        print(f"Battery station changed to: ({world_x:.1f}, {world_y:.1f})")


if __name__ == "__main__":
    print("\nTesting Drone2dBudgetEnv...")
    env_budget = Drone2dBudgetEnv(render_mode='human', change_target=True, change_battery_station=True)
    obs, info = env_budget.reset(seed=43)
    terminated = truncated = False
    total_reward_budget = 0
    for _ in range(500): # Longer loop to test battery
        action = env_budget.action_space.sample()
        obs, reward, terminated, truncated, info = env_budget.step(action)
        total_reward_budget += reward
        if terminated or truncated: break
    env_budget.close()
    print(f"Budget Env Test Done. Final Info: {info}")
