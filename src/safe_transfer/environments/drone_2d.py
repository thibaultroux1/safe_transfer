import os
import sys
import random
from typing import Dict, Any, Tuple, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, MOUSEBUTTONUP, BUTTON_LEFT)
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

from safe_transfer.environments.drone import Drone
from safe_transfer.environments.utils import clip_norm, distance_normalized


class Drone2dEnv(gym.Env):
    """
    Base 2D Drone Environment using PyMunk and Pygame.

    Observation Space (normalized):
        - velocity_x
        - velocity_y
        - angular_velocity (omega)
        - angle (alpha)
        - drone_pos_x
        - drone_pos_y
        - target_pos_x
        - target_pos_y

    Action Space (normalized): [-1, 1] for left and right motor thrust ratio.

    Args:
        render_sim (bool): If True, render the simulation using Pygame.
        render_path (bool): If True, draw the drone's path.
        render_shade (bool): If True, draw drone shades.
        shade_distance (int): Distance threshold to draw a new shade.
        n_steps (int): Maximum number of steps per episode.
        n_fall_steps (int): Initial steps where the drone just falls (optional stabilization).
        change_target (bool): Allow changing target with left mouse click in render mode.
        random_change_target (int): If not None, inverse of the probability of the target randomly changing location (even if not reached)
        initial_throw (bool): Apply random initial force/torque to the drone.
        seed (int): Random seed.
        window_size (int): Simulation window size (width and height).
        max_velocity_component (float): Used for normalizing velocity observations.
        max_angular_velocity (float): Used for normalizing angular velocity observation.
        thrust_force_scale (float): Scaling factor for converting action [-1, 1] to force.
        target_reach_threshold (float): Normalized distance threshold to consider target reached.
        reward_distance_beta (float): Parameter for exponential reward based on distance.
        reward_goal (float): Bonus reward for reaching target.
        penalty_crash (float): Penalty for crashing or going out of bounds.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self,
                 render_mode: str | None = None, # Standard Gym API
                 render_path=True,
                 render_shade=False,
                 shade_distance=70,
                 n_steps=2000,
                 n_fall_steps=10,
                 change_target=True,
                 random_change_target: int | None=None,
                 initial_throw=True,
                 seed=555,
                 window_size=800,
                 max_velocity_component=1330.0,
                 max_angular_velocity=11.7,
                 thrust_force_scale=1000.0,
                 target_reach_threshold=0.025,
                 reward_distance_beta=3.0,
                 reward_goal=30.0,
                 penalty_crash=-100.0,
                 ):

        super().__init__()

        random.seed(seed)
        np.random.seed(seed)

        self.window_size = window_size
        self.render_mode = render_mode
        self.render_path = render_path if self.render_mode == "human" else False
        self.render_shade = render_shade if self.render_mode == "human" else False
        self.shade_distance_threshold = shade_distance

        self.max_time_steps = n_steps
        self.stabilisation_delay = n_fall_steps
        self.change_target = change_target
        self.random_change_target = random_change_target
        self.initial_throw = initial_throw

        self.info_keys = ["crash", "truncated", "goals_reached", "task_completed", "flip", "out_of_bounds"]

        # Physics/Simulation parameters
        self.max_velocity_component = max_velocity_component
        self.max_angular_velocity = max_angular_velocity
        self.force_scale = thrust_force_scale
        self._dt = 1.0 / self.metadata["render_fps"]

        # Reward/Termination parameters
        self.target_reach_threshold = target_reach_threshold
        self.reward_distance_beta = reward_distance_beta
        self.reward_goal = reward_goal
        self.penalty_crash = penalty_crash

        # Pygame/Pymunk objects (initialized in reset/render)
        self.screen = None
        self.clock = None
        self.font = None
        self.shade_image = None
        self.space = None
        self.draw_options = None
        self.drone: Drone | None = None
        self.drone_radius = 0 # Set after drone creation

        # State variables (reset properly)
        self._first_step = True
        self._current_time_step = 0
        self._left_force_val = 0.0
        self._right_force_val = 0.0
        self._flight_path: List[Tuple[float, float]] = []
        self._drop_path: List[Tuple[float, float]] = []
        self._path_drone_shade: List[List[float]] = []
        self._last_shade_pos = (0.0, 0.0)
        self.reward_value = 0.0 # For rendering

        # Target position (raw coordinates)
        self.x_target = 0.0
        self.y_target = 0.0

        # Define action space
        min_action = np.array([-1, -1], dtype=np.float32)
        max_action = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        # Define observation space (will be finalized in _init_observation_space)
        self._init_observation_space()

        if self.render_mode == "human":
            self._init_pygame()

    def _init_observation_space(self):
        """Initializes the base observation space."""
        num_obs = 8 # vel_x, vel_y, omega, alpha, pos_x, pos_y, target_x, target_y
        min_observation = np.array([-1] * num_obs, dtype=np.float32)
        max_observation = np.array([1] * num_obs, dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)

    def _init_pygame(self):
        """Initializes Pygame for rendering."""
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Drone2d Environment")
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            pymunk.pygame_util.positive_y_is_up = True # Crucial for Pymunk rendering

            # Initialize font
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 24)

            # Load images (handle potential path issues)
            try:
                script_dir = os.path.dirname(__file__)
                icon_path = os.path.join(script_dir, "img", "icon.png")
                pygame.display.set_icon(pygame.image.load(icon_path))
                img_path = os.path.join(script_dir, "img", "shade.png")
                self.shade_image = pygame.image.load(img_path)
            except pygame.error as e:
                print(f"Warning: Could not load images. Pygame error: {e}")
            except FileNotFoundError as e:
                 print(f"Warning: Could not find image files. Error: {e}")

    def _init_pymunk(self):
        """Initializes Pymunk space and drone."""
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -1000) # Standard gravity

        if self.render_mode == "human" and self.screen:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

        # Generate drone's starting position
        margin = 200 # Avoid starting too close to edges
        random_x = random.uniform(margin, self.window_size - margin)
        random_y = random.uniform(margin, self.window_size - margin)
        angle_rand = random.uniform(-np.pi / 4, np.pi / 4)

        # Create Drone instance (adjust parameters as needed)
        self.drone = Drone(random_x, random_y, angle_rand, height=20, width=100,
                           mass_f=0.2, mass_l=0.4, mass_r=0.4, space=self.space)
        self.drone_radius = self.drone.drone_radius

    def _generate_target(self):
        """Generates a random target position within bounds."""
        margin = 100 # Keep target away from edges
        self.x_target = random.uniform(margin, self.window_size - margin)
        self.y_target = random.uniform(margin, self.window_size - margin)

    def _get_drone_state(self) -> Tuple[float, float, float, float, float, float]:
        """Helper to get raw drone physics state."""
        body = self.drone.frame_shape.body
        pos_x, pos_y = body.position
        vel_x, vel_y = body.velocity
        omega = body.angular_velocity
        alpha = body.angle # Angle in radians
        # Normalize angle to [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        return pos_x, pos_y, vel_x, vel_y, omega, alpha

    def get_observation(self) -> np.ndarray:
        """Computes the observation vector."""
        if self.drone is None or self.drone.frame_shape.body is None:
             # Return a default observation if drone doesn't exist yet (e.g., before first reset)
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        pos_x, pos_y, vel_x, vel_y, omega, alpha = self._get_drone_state()

        # Normalize observations
        norm_vel_x = np.clip(vel_x / self.max_velocity_component, -1, 1)
        norm_vel_y = np.clip(vel_y / self.max_velocity_component, -1, 1)
        norm_omega = np.clip(omega / self.max_angular_velocity, -1, 1)
        norm_alpha = np.clip(alpha / (np.pi / 2), -1, 1) # Normalize angle based on full rotation possibility
        norm_pos_x = clip_norm(pos_x, self.window_size)
        norm_pos_y = clip_norm(pos_y, self.window_size)
        norm_target_x = clip_norm(self.x_target, self.window_size)
        norm_target_y = clip_norm(self.y_target, self.window_size)

        return np.array([
            norm_vel_x, norm_vel_y, norm_omega, norm_alpha,
            norm_pos_x, norm_pos_y, norm_target_x, norm_target_y
        ], dtype=np.float32)

    def _apply_action_and_step_physics(self, action: np.ndarray):
        """Applies action forces and steps the Pymunk simulation."""
        # Map action [-1, 1] to force [0, force_scale]
        # Ensure action is numpy array
        action = np.asarray(action)
        self._left_force_val = (action[0] / 2 + 0.5) * self.force_scale
        self._right_force_val = (action[1] / 2 + 0.5) * self.force_scale

        # Apply forces at motor positions relative to the drone's frame center
        # Forces are applied in the direction the drone is currently facing (local y-axis)
        force_vec_left = Vec2d(0, self._left_force_val)
        force_vec_right = Vec2d(0, self._right_force_val)
        motor_pos_left = Vec2d(-self.drone_radius, 0)
        motor_pos_right = Vec2d(self.drone_radius, 0)

        self.drone.frame_shape.body.apply_force_at_local_point(force_vec_left, motor_pos_left)
        self.drone.frame_shape.body.apply_force_at_local_point(force_vec_right, motor_pos_right)

        # Step the physics simulation
        self.space.step(self._dt)
        self._current_time_step += 1

    def _update_paths_and_shades(self):
        """Updates rendering artifacts like paths and shades."""
        if self.render_mode != "human": return

        pos_x, pos_y, _, _, _, angle = self._get_drone_state()
        screen_pos = (pos_x, self.window_size - pos_y) # Convert to Pygame coordinates

        # --- Path Recording ---
        if self._first_step:
            if self.render_path:
                self._add_position_to_path(self._drop_path, screen_pos) # First point for drop path
                self._add_position_to_path(self._flight_path, screen_pos) # First point for flight path
        else:
             if self.render_path:
                 # Add current position to flight path if not stabilization phase
                 if self._current_time_step > self.stabilisation_delay:
                     self._add_position_to_path(self._flight_path, screen_pos)
                 else:
                     # Otherwise add to drop path
                      self._add_position_to_path(self._drop_path, screen_pos)


        # --- Shade Recording ---
        if self.render_shade:
            current_pos = (pos_x, pos_y)
            dist_sq = (pos_x - self._last_shade_pos[0])**2 + (pos_y - self._last_shade_pos[1])**2
            if not self._first_step and dist_sq > self.shade_distance_threshold**2:
                self._add_drone_shade(pos_x, pos_y, angle)
                self._last_shade_pos = current_pos


    def _calculate_reward(self, obs: np.ndarray) -> float:
        """Calculates the reward based on the current observation."""
        # Unpack necessary components from base observation
        norm_pos_x, norm_pos_y, norm_target_x, norm_target_y = obs[4:8]

        dist_target = distance_normalized(norm_pos_x, norm_pos_y, norm_target_x, norm_target_y)

        # Exponential reward inversely proportional to distance
        reward = np.exp(-self.reward_distance_beta * dist_target)

        # Bonus for reaching target (optional, if change_target is True)
        if self.change_target and dist_target <= self.target_reach_threshold:
            self._generate_target() # Generate new target immediately
            self.info['goals_reached'] += 1
            self.info['task_completed'] += 1
            reward += self.reward_goal # Large bonus for reaching the goal
        
        # Change target randomly with a certain probability if random_change_target is set
        if self.random_change_target is not None and random.random() < 1 / self.random_change_target:
            self._generate_target()

        return float(reward)

    def _check_termination(self, obs: np.ndarray) -> Tuple[bool, bool]:
        """Checks for termination conditions (crash, out of bounds, flipped)."""
        # Unpack necessary components
        norm_alpha, norm_pos_x, norm_pos_y = obs[3:6]

        # Crash conditions
        is_out_of_bounds = abs(norm_pos_x) >= 1.0 or abs(norm_pos_y) >= 1.0
        # Angle normalization means |alpha| ~ 1 means flipped pi radians (180 deg)
        is_flipped = abs(norm_alpha) >= 1 # Check if close to +/- pi/2

        terminated = False
        if is_out_of_bounds:
            self.info['crash'] += 1
            self.info['out_of_bounds'] +=1
            terminated = True
        elif is_flipped:
            self.info['crash'] += 1
            self.info['flip'] += 1
            terminated = True

        # Truncation condition (time limit)
        truncated = self._current_time_step >= self.max_time_steps
        if truncated:
            self.info["truncated"] = 1

        return terminated, truncated


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Runs one timestep of the environment's dynamics."""
        self._apply_action_and_step_physics(action)
        self._update_paths_and_shades() # Update paths before checking state

        if self._first_step: self._first_step = False

        obs = self.get_observation()
        reward = self._calculate_reward(obs)
        terminated, truncated = self._check_termination(obs)

        # Apply penalty for termination (crash)
        if terminated:
            reward = self.penalty_crash

        self.reward_value = reward # Store for rendering

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self.info


    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        # Reset state variables
        self._first_step = True
        self._current_time_step = 0
        self._left_force_val = 0.0
        self._right_force_val = 0.0
        self._flight_path = []
        self._drop_path = []
        self._path_drone_shade = []
        self._last_shade_pos = (0.0, 0.0)
        self.reward_value = 0.0
        self.info = {k: 0 for k in self.info_keys} # Base info dict

        # Re-initialize Pymunk and Pygame drawing if needed
        self._init_pymunk()
        if self.render_mode == "human":
            self._init_pygame() # Ensure pygame is ready
            if self.draw_options is None and self.screen: # Recreate draw options if needed
                 self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # Generate initial target
        self._generate_target()

        # Optional: Apply initial throw and stabilization delay
        if self.initial_throw:
            self._apply_initial_throw()
        self._apply_stabilization_delay()

        # Need to record first position after potential throw/stabilization
        if self.render_mode == "human":
            pos_x, pos_y, _, _, _, angle = self._get_drone_state()
            screen_pos = (pos_x, self.window_size - pos_y)
            if self.render_path:
                self._add_position_to_path(self._drop_path, screen_pos)
                self._add_position_to_path(self._flight_path, screen_pos)
            if self.render_shade:
                self._add_drone_shade(pos_x, pos_y, angle)
                self._last_shade_pos = (pos_x, pos_y)

        observation = self.get_observation()
        return observation, self.info

    def _apply_initial_throw(self):
        """Applies a random initial force and torque."""
        if self.drone is None: return
        body = self.drone.frame_shape.body

        throw_angle = random.uniform(0, 2 * np.pi)
        throw_force_mag = random.uniform(0, 25000) # Original value
        throw_force = Vec2d(np.cos(throw_angle) * throw_force_mag, np.sin(throw_angle) * throw_force_mag)
        body.apply_force_at_world_point(throw_force * self._dt, body.position) # Use impulse

        throw_rotation_mag = random.uniform(-3000, 3000) # Original value
        # Apply as torque impulse directly
        body.apply_force_at_local_point(Vec2d(0, throw_rotation_mag * self._dt), (-self.drone_radius, 0))
        body.apply_force_at_local_point(Vec2d(0, -throw_rotation_mag * self._dt), (self.drone_radius, 0))

        # Step simulation once to let the throw take effect before stabilization
        self.space.step(self._dt)
        # Don't increment timestep here, it's part of setup

    def _apply_stabilization_delay(self):
        """Runs the simulation for a few steps without agent control."""
        for _ in range(self.stabilisation_delay):
            # No action applied, just gravity and existing momentum
            self.space.step(self._dt)
            # Update paths during stabilization if rendering
            self._update_paths_and_shades()
        # Don't increment timestep here, it's part of setup

    # --- Rendering ---

    def render(self) -> None:
        """Renders the environment by calling helper methods."""
        if self.render_mode != "human":
            return
        if self.screen is None or self.clock is None or self.font is None or self.space is None or self.drone is None or self.draw_options is None:
            self._init_pygame() # Try to initialize if not already
            if self.screen is None: return

        # --- Event Handling & Background ---
        self._handle_pygame_events()
        self._render_background()

        # --- Draw Simulation Elements ---
        self._render_environment_elements()

        # --- Draw HUD/Overlay Elements ---
        self._render_hud_elements()

        # --- Finalize Frame ---
        self._finalize_render()

    def _render_background(self):
        """Renders the static background and borders."""
        self.screen.fill((243, 243, 243)) # Light grey background
        # Decorative borders
        border_color_outer = (24, 114, 139)
        border_color_inner1 = (33, 158, 188)
        border_color_inner2 = (142, 202, 230)
        pygame.draw.rect(self.screen, border_color_outer, pygame.Rect(0, 0, self.window_size, self.window_size), 8)
        pygame.draw.rect(self.screen, border_color_inner1, pygame.Rect(100, 100, self.window_size-200, self.window_size-200), 4)
        pygame.draw.rect(self.screen, border_color_inner2, pygame.Rect(200, 200, self.window_size-400, self.window_size-400), 4)

    def _render_environment_elements(self):
        """Renders the drone, paths, target, forces, etc."""
        if self.screen is None or self.space is None or self.drone is None or self.draw_options is None:
             return

        # --- Drawing Shades ---
        self._render_shades()

        # --- Drawing Pymunk Objects (Drone) ---
        self.space.debug_draw(self.draw_options)

        # --- Drawing Paths ---
        self._render_paths()

        # --- Drawing Target ---
        self._render_target()

        # --- Drawing Forces ---
        self._render_motor_forces()

    # --- Renders HUD elements like text ---
    def _render_hud_elements(self):
        """Renders overlay elements like text info."""
        if self.screen is None or self.font is None:
            return

        # --- Drawing Text Info ---
        # self._render_text_info() # Call the existing text rendering method

    # --- Finalizes the frame ---
    def _finalize_render(self):
        """Handles final display flip and clock tick."""
        if self.render_mode == "human" and self.screen and self.clock:
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    # --- Existing helper methods used by the above ---
    def _handle_pygame_events(self):
        """Handles basic Pygame events."""
        if not self.render_mode == "human": return

        for event in pygame.event.get():
            if event.type == QUIT:
                self.close()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self.close()
                sys.exit()
            elif event.type == MOUSEBUTTONUP:
                 if event.button == BUTTON_LEFT:
                     x, y = event.pos
                     self.change_target_point(x, self.window_size - y) # Convert y coord

    def _render_shades(self):
        """Renders drone shades."""
        if not (self.render_shade and self.shade_image): return
        for shade_data in self._path_drone_shade:
            x, y, angle_rad = shade_data
            angle_deg = np.degrees(angle_rad) # Pygame rotate uses degrees
            try:
                rotated_image = pygame.transform.rotate(self.shade_image, angle_deg) # Pygame rotation is clockwise
                rect = rotated_image.get_rect(center=(x, self.window_size - y)) # Convert y
                self.screen.blit(rotated_image, rect)
            except pygame.error as e:
                 print(f"Warning: Failed to rotate/blit shade image. Error: {e}")
                 break

    def _render_paths(self):
        """Renders drone flight and drop paths."""
        if not self.render_path: return
        if len(self._flight_path) > 1:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self._flight_path) # Blue flight path
        if len(self._drop_path) > 1:
            pygame.draw.aalines(self.screen, (255, 0, 0), False, self._drop_path) # Red drop/stabilization path

    def _render_target(self):
        """Renders the target marker."""
        if not self.drone: return
        target_screen_x = int(self.x_target)
        target_screen_y = int(self.window_size - self.y_target) # Convert y
        pygame.draw.circle(self.screen, (255, 0, 0), (target_screen_x, target_screen_y), 5) # Red circle

    def _render_motor_forces(self):
        """Renders lines indicating motor thrust."""
        if not (self.drone and self.drone.frame_shape): return

        body = self.drone.frame_shape.body
        vector_scale = 0.05
        max_force_len = self.force_scale * vector_scale

        # Left Motor
        motor_pos_world_l = body.local_to_world((-self.drone_radius, 0))
        force_end_max_local_l = Vec2d(-self.drone_radius, max_force_len)
        force_end_curr_local_l = Vec2d(-self.drone_radius, self._left_force_val * vector_scale)
        force_end_max_world_l = body.local_to_world(force_end_max_local_l)
        force_end_curr_world_l = body.local_to_world(force_end_curr_local_l)
        start_screen_l = (int(motor_pos_world_l.x), int(self.window_size - motor_pos_world_l.y))
        max_end_screen_l = (int(force_end_max_world_l.x), int(self.window_size - force_end_max_world_l.y))
        curr_end_screen_l = (int(force_end_curr_world_l.x), int(self.window_size - force_end_curr_world_l.y))
        pygame.draw.line(self.screen, (179, 179, 179), start_screen_l, max_end_screen_l, 4)
        pygame.draw.line(self.screen, (255, 0, 0), start_screen_l, curr_end_screen_l, 4)

        # Right Motor
        motor_pos_world_r = body.local_to_world((self.drone_radius, 0))
        force_end_max_local_r = Vec2d(self.drone_radius, max_force_len)
        force_end_curr_local_r = Vec2d(self.drone_radius, self._right_force_val * vector_scale)
        force_end_max_world_r = body.local_to_world(force_end_max_local_r)
        force_end_curr_world_r = body.local_to_world(force_end_curr_local_r)
        start_screen_r = (int(motor_pos_world_r.x), int(self.window_size - motor_pos_world_r.y))
        max_end_screen_r = (int(force_end_max_world_r.x), int(self.window_size - force_end_max_world_r.y))
        curr_end_screen_r = (int(force_end_curr_world_r.x), int(self.window_size - force_end_curr_world_r.y))
        pygame.draw.line(self.screen, (179, 179, 179), start_screen_r, max_end_screen_r, 4)
        pygame.draw.line(self.screen, (255, 0, 0), start_screen_r, curr_end_screen_r, 4)

    def _render_text_info(self):
        """Renders text information like reward, distance, position."""
        if not self.font or not self.drone: return

        reward_text = self.font.render(f'Reward: {self.reward_value:.2f}', True, (0, 0, 0))
        self.screen.blit(reward_text, (10, 10))

        obs = self.get_observation()
        # Ensure observation has enough elements before accessing them
        if len(obs) >= 8:
            norm_pos_x, norm_pos_y, norm_target_x, norm_target_y = obs[4:8]
            dist_target = distance_normalized(norm_pos_x, norm_pos_y, norm_target_x, norm_target_y)

            distance_text = self.font.render(f'Dist (N): {dist_target:.2f}', True, (0, 0, 0))
            self.screen.blit(distance_text, (10, 40)) # Adjusted y position slightly

            position_text = self.font.render(f'Pos (N): ({norm_pos_x:.2f}, {norm_pos_y:.2f})', True, (0, 0, 0))
            self.screen.blit(position_text, (10, 70)) # Adjusted y position slightly
        else:
            # Handle case where observation might not be fully formed yet (e.g., during init)
            err_text = self.font.render('Obs N/A', True, (255, 0, 0))
            self.screen.blit(err_text, (10, 40))


    def close(self):
        """Closes the environment and cleans up Pygame resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
            print("Pygame closed.")


    # --- Path/Shade Helpers ---
    def _add_position_to_path(self, path_list: List[Tuple[float, float]], screen_pos: Tuple[float, float]):
        """Adds a point to a path list, converting to integer coordinates."""
        path_list.append((int(screen_pos[0]), int(screen_pos[1])))

    def _add_drone_shade(self, x: float, y: float, angle_rad: float):
        """Adds drone shade data."""
        self._path_drone_shade.append([x, y, angle_rad])

    # --- Point Manipulation Helpers ---
    def change_target_point(self, world_x: float, world_y: float):
        """Sets the target position."""
        self.x_target = world_x
        self.y_target = world_y
        print(f"Target changed to: ({world_x:.1f}, {world_y:.1f})")

    def distance_drone_to_world(self, world_x: float, world_y: float) -> float:
        """Calculates distance from drone center to a world point."""
        if not self.drone: return float('inf')
        drone_x, drone_y = self.drone.frame_shape.body.position
        return np.sqrt((drone_x - world_x)**2 + (drone_y - world_y)**2)

    def distance_drone_to_normalized(self, norm_x: float, norm_y: float) -> float:
        """Calculates distance from drone center (normalized) to a normalized point."""
        obs = self.get_observation()
        drone_norm_x, drone_norm_y = obs[4:6]
        return distance_normalized(drone_norm_x, drone_norm_y, norm_x, norm_y)


if __name__ == '__main__':
    # --- Test Base Env ---
    print("\nTesting Drone2dEnv...")
    env_base = Drone2dEnv(render_mode='human', change_target=True)
    obs, info = env_base.reset(seed=42)
    terminated = truncated = False
    total_reward_base = 0
    for _ in range(500): # Short test loop
        action = env_base.action_space.sample()
        obs, reward, terminated, truncated, info = env_base.step(action)
        total_reward_base += reward
        if terminated or truncated: break
    env_base.close()
    print(f"Base Env Test Done. Final Info: {info}")
