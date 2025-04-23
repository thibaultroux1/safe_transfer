import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import pygame # Keep pygame import for event handling in main loop
from typing import List, Dict, Any, Tuple

import safe_transfer.environments
from safe_transfer.environments.drone_2d_budget import Drone2dBudgetEnv
from safe_transfer.environments.drone_2d_budget_package import Drone2dBudgetPackageEnv
from safe_transfer.nn import AgentContinuous as Agent


class DummyEnvWrapper:
    def __init__(self, obs_space, action_space):
        self.single_observation_space = obs_space
        self.single_action_space = action_space


class Drone2dBudgetHierarchyEnv(Drone2dBudgetEnv):
    """
    Hierarchical wrapper for Drone2dBudgetEnv.
    High-level action selects a low-level policy (GoToTarget or GoToBattery).
    """
    def __init__(self,
                 low_level_policies_paths: List[str],
                 env_ids: List[str], # IDs of envs policies were trained on (for space info)
                 render_mode: str | None = None,
                 **kwargs): # Pass other args (battery params, n_steps etc.) to base
        """
        Args:
            low_level_policies_paths: List of paths to saved low-level policy checkpoints.
                                      Expected order: [policy_goto_target, policy_goto_battery]
            env_ids: List of gym environment IDs corresponding to the training environments
                     for each policy in low_level_policies_paths. Used to get space info.
                     Example: ["Drone2dEnv-v1", "Drone2dEnv-v1"] if both trained on base env.
            render_mode: Rendering mode for the base environment.
            **kwargs: Additional arguments passed to Drone2dBudgetEnv constructor
                      (e.g., initial_battery, n_steps, seed).
        """
        # Initialize the base Budget environment
        super().__init__(render_mode=render_mode, **kwargs)
        if not env_ids or len(env_ids) != len(low_level_policies_paths):
            raise ValueError("env_ids must be a list of environment IDs corresponding to low-level policies.")

        try:
            dummy_env = gym.make(env_ids[0])
            self.low_level_observation_space = dummy_env.observation_space
            self.low_level_action_space = dummy_env.action_space
            dummy_env.close()
            self.dummy_env_for_agent = DummyEnvWrapper(self.low_level_observation_space, self.low_level_action_space)
        except Exception as e:
            raise RuntimeError(f"Could not create dummy environment '{env_ids[0]}' to get space info. "
                               f"Ensure this env ID is registered and compatible. Error: {e}")

        self.low_level_policies = self._load_agents(low_level_policies_paths)
        self.number_low_level_policies = len(self.low_level_policies)

        # High-level action space: Choose which low-level policy to use
        self.action_space = spaces.Discrete(self.number_low_level_policies)

        # Store previous high-level action for rendering or analysis
        self.previous_high_level_action = 0

    def _load_agents(self, policy_paths: List[str]) -> List[Agent]:
        """Loads low-level policies."""
        agents = []
        for i, path in enumerate(policy_paths):

                agent = Agent(self.dummy_env_for_agent)

                # Specify map_location for CPU loading if model was saved on GPU
                checkpoint = torch.load(path, map_location=torch.device('cpu'))
                # Check common key names for the state dict
                state_dict_key = "model_state_dict"
                if state_dict_key not in checkpoint:
                    raise KeyError(f"Could not find key '{state_dict_key}' or recognize state dict in checkpoint: {path}")
                else:
                    state_dict = checkpoint[state_dict_key]

                agent.load_state_dict(state_dict)
                agent.eval() # Set agent to evaluation mode
                agents.append(agent)
                print(f"Loaded low-level policy {i} from: {path}")
        return agents

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Takes a high-level action (index of low-level policy), gets a low-level
        action from that policy, and steps the underlying environment.
        """
        if action < 0 or action >= self.number_low_level_policies:
            raise ValueError(f"Invalid high-level action: {action}. Must be between 0 and {self.number_low_level_policies - 1}.")

        self.previous_high_level_action = action
        chosen_low_level_policy = self.low_level_policies[action]

        # Get observation formatted for the low-level policy
        low_level_obs = self._get_low_level_observation(action)
        # Ensure observation is float32, add batch dimension
        low_level_obs_tensor = torch.tensor(low_level_obs, dtype=torch.float32).unsqueeze(0)

        # Get action from the selected low-level policy
        with torch.no_grad():
            low_level_action_tensor, _, _, _ = chosen_low_level_policy.get_action_and_value(low_level_obs_tensor)

        # Clip action to valid range [-1, 1] and convert to numpy
        low_level_action = low_level_action_tensor[0].cpu().numpy().clip(-1.0, 1.0)

        # Step the *base* environment (Drone2dBudgetEnv) using the low-level action
        # super().step() calls Drone2dBudgetEnv.step()
        # It returns the *full* observation (11 elements), reward, term, trunc, info
        obs, reward, terminated, truncated, info = super().step(low_level_action)

        # The observation returned by the *hierarchical* env's step is the *full*
        # observation from the underlying env. The high-level policy (if any)
        # would operate on this full observation.
        return obs, reward, terminated, truncated, info

    def _get_low_level_observation(self, high_level_action: int) -> np.ndarray:
        """
        Constructs the 8-element observation expected by the low-level policies.
        """
        # Get the full 11-element observation from the base environment
        full_obs = super().get_observation()
        # Unpack carefully based on the NEW Drone2dBudgetEnv observation structure:
        # [vel_x, vel_y, omega, alpha, pos_x, pos_y, target_x, target_y, battery_x, battery_y, battery_ratio]
        velocity_x = full_obs[0]
        velocity_y = full_obs[1]
        omega = full_obs[2]
        alpha = full_obs[3]
        pos_x = full_obs[4]
        pos_y = full_obs[5]
        orig_target_x = full_obs[6]
        orig_target_y = full_obs[7]
        pos_x_battery = full_obs[8]
        pos_y_battery = full_obs[9]
        # battery_ratio = full_obs[10] # Low-level policy does not need this

        # Determine the effective target for the low-level policy
        if high_level_action == 0: # Policy 0: Go to original target
            effective_target_x = orig_target_x
            effective_target_y = orig_target_y
        elif high_level_action == 1: # Policy 1: Go to battery
            effective_target_x = pos_x_battery
            effective_target_y = pos_y_battery
        else:
            # Handle cases with more policies if needed
            print(f"Warning: Unexpected high-level action {high_level_action} in Drone2dBudgetHierarchyEnv._get_low_level_observation. Defaulting target.")
            effective_target_x = orig_target_x
            effective_target_y = orig_target_y

        # Construct the 8-element observation array in the order
        # the low-level policies were trained on.
        low_level_observation = np.array([
            velocity_x,
            velocity_y,
            omega,
            alpha,
            pos_x,
            pos_y,
            effective_target_x,
            effective_target_y
        ], dtype=np.float32) # Ensure correct dtype

        # Sanity check the shape
        if low_level_observation.shape[0] != self.low_level_observation_space.shape[0]:
             raise ValueError(f"Constructed low-level observation shape {low_level_observation.shape} "
                              f"does not match expected shape {self.low_level_observation_space.shape}")

        return low_level_observation

    # Rendering is inherited from Drone2dBudgetEnv and should work.
    # We might add visualization for the high-level action later if needed.
    def render(self):
        """ Renders the environment, potentially adding high-level action display. """
        super().render()

# --- Package Hierarchy ---

class Drone2dBudgetPackageHierarchyEnv(Drone2dBudgetHierarchyEnv, Drone2dBudgetPackageEnv):
    """
    Hierarchical wrapper for Drone2dBudgetPackageEnv.
    High-level action selects low-level policy (GoToTarget, GoToBattery, GoToPackage).
    """
    def __init__(self,
                 low_level_policies_paths: List[str],
                 env_ids: List[str],
                 render_mode: str | None = None,
                 **kwargs):
        """
        Args:
            low_level_policies_paths: List of paths. Expected order:
                                      [policy_goto_target, policy_goto_battery, policy_goto_package]
            env_ids: List of gym env IDs corresponding to policy training envs.
            render_mode: Rendering mode.
            **kwargs: Additional arguments for Drone2dBudgetPackageEnv/Drone2dBudgetEnv.
        """
        super().__init__(
            low_level_policies_paths=low_level_policies_paths,
            env_ids=env_ids,
            render_mode=render_mode,
            **kwargs
        )


        self.action_space = spaces.Discrete(self.number_low_level_policies) # Ensure it's set correctly


    def _get_low_level_observation(self, high_level_action: int) -> np.ndarray:
        """
        Constructs the 8-element observation expected by the low-level policies,
        using the full 14-element state from Drone2dBudgetPackageEnv.
        """
        # Get the full 14-element observation from Drone2dBudgetPackageEnv
        full_obs = super().get_observation()
        # Unpack based on the NEW Drone2dBudgetPackageEnv observation structure:
        velocity_x = full_obs[0]
        velocity_y = full_obs[1]
        omega = full_obs[2]
        alpha = full_obs[3]
        pos_x = full_obs[4]
        pos_y = full_obs[5]
        orig_target_x = full_obs[6]
        orig_target_y = full_obs[7]
        pos_x_battery = full_obs[8]
        pos_y_battery = full_obs[9]
        # battery_ratio = full_obs[10]
        pos_x_package = full_obs[11]
        pos_y_package = full_obs[12]
        # package_collected = full_obs[13] # Low-level policy likely doesn't need this flag

        # Determine the effective target for the low-level policy
        if high_level_action == 0: # Policy 0: Go to original target
            effective_target_x = orig_target_x
            effective_target_y = orig_target_y
        elif high_level_action == 1: # Policy 1: Go to battery
            effective_target_x = pos_x_battery
            effective_target_y = pos_y_battery
        elif high_level_action == 2: # Policy 2: Go to package
            effective_target_x = pos_x_package
            effective_target_y = pos_y_package
        else:
            print(f"Warning: Unexpected high-level action {high_level_action} in Drone2dBudgetPackageHierarchyEnv._get_low_level_observation. Defaulting target.")
            effective_target_x = orig_target_x
            effective_target_y = orig_target_y

        # Construct the 8-element observation array
        low_level_observation = np.array([
            velocity_x,
            velocity_y,
            omega,
            alpha,
            pos_x,
            pos_y,
            effective_target_x,
            effective_target_y
        ], dtype=np.float32)

        # Sanity check shape
        if low_level_observation.shape[0] != self.low_level_observation_space.shape[0]:
             raise ValueError(f"Constructed low-level observation shape {low_level_observation.shape} "
                              f"does not match expected shape {self.low_level_observation_space.shape}")

        return low_level_observation

    # --- Rendering ---

    def render(self):
        """ Renders the Package env and adds high-level action text. """
        self._handle_pygame_events()
        self._render_background()
        self._render_environment_elements()
        self._render_hud_elements()
        # if self.font:
            #  self._render_high_level_action() # Call the helper
        self._finalize_render()


    def _render_high_level_action(self):
        """Helper to draw text indicating the current high-level action."""
        action_names = {0: "Go to Goal", 1: "Go to Battery", 2: "Go to Package"}
        action_name = action_names.get(self.previous_high_level_action, f"Action {self.previous_high_level_action}")

        y_pos = 130
        action_text = self.font.render(f'HL Action: {action_name}', True, (0, 0, 0))
        self.screen.blit(action_text, (10, y_pos))

    def _render_feature_importance(self, feature_importances):
         feature_names = ["vel_x", "vel_y", "omega", "alpha", "pos_x", "pos_y", "target_x", "target_y", "battery_x", "battery_y", "bat_ratio", "package_x", "package_y", "pack_coll"]
         start_x, start_y = 10, 160
         line_height = 20
         max_importance_len = 20

         if len(feature_importances) > len(feature_names):
              print(f"Warning: More feature importances ({len(feature_importances)}) than names ({len(feature_names)}). Truncating.")
              feature_importances = feature_importances[:len(feature_names)]
         elif len(feature_importances) < len(feature_names):
              print(f"Warning: Fewer feature importances ({len(feature_importances)}) than names ({len(feature_names)}). Padding names.")
              feature_names = feature_names[:len(feature_importances)]


         for i, importance in enumerate(feature_importances):
             blocks = '█' * int(min(max(importance, 0), 1) * max_importance_len) # Ensure importance is [0,1]
             feature_text = self.font.render(f"{feature_names[i]}: {blocks} ({importance:.2f})", True, (0, 0, 0))
             self.screen.blit(feature_text, (start_x, start_y + i * line_height))


if __name__ == '__main__':

    POLICY_GOTO_TARGET = ""
    POLICY_GOTO_BATTERY = ""
    POLICY_GOTO_PACKAGE = ""

    LOW_LEVEL_ENV_ID = "Drone2D-v0"

    # --- Test Budget Hierarchy ---
    print("--- Testing Budget Hierarchy ---")
    try:
        env_budget_h = Drone2dBudgetHierarchyEnv(
            low_level_policies_paths=[POLICY_GOTO_TARGET, POLICY_GOTO_BATTERY],
            env_ids=[LOW_LEVEL_ENV_ID] * 2, # Provide ID for each policy path
            render_mode="human",
            n_steps=2000,
            seed=0
        )

        obs, info = env_budget_h.reset(seed=0)
        terminated = truncated = False
        total_reward = 0
        running = True
        selected_action = 0 # Default: Go to goal

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    elif event.key == pygame.K_g: selected_action = 0; print("HL Action: Go To Goal")
                    elif event.key == pygame.K_b: selected_action = 1; print("HL Action: Go To Battery")

            if not running: break

            obs, reward, terminated, truncated, info = env_budget_h.step(selected_action)
            total_reward += reward

            if terminated or truncated:
                print(f"Budget Hierarchy Episode finished! Total reward: {total_reward:.2f}, Info: {info}")
                total_reward = 0
                obs, info = env_budget_h.reset() # Reset for next episode test
                terminated = truncated = False # Continue running after reset

        env_budget_h.close()
        print("Budget Hierarchy Test Done.")

    except (FileNotFoundError, RuntimeError, ValueError, Exception) as e:
        print(f"\nERROR setting up/running Budget Hierarchy Test: {e}")
        print("Please ensure policy paths and env IDs are correct.")


    # --- Test Package Hierarchy ---
    print("\n--- Testing Package Hierarchy ---")
    try:
        env_package_h = Drone2dBudgetPackageHierarchyEnv(
            low_level_policies_paths=[POLICY_GOTO_TARGET, POLICY_GOTO_BATTERY, POLICY_GOTO_PACKAGE],
            env_ids=[LOW_LEVEL_ENV_ID] * 3, # Provide ID for each policy path
            render_mode="human",
            n_steps=2000,
            seed=0,
        )

        obs, info = env_package_h.reset(seed=0)
        terminated = truncated = False
        total_reward = 0
        running = True
        selected_action = 2 # Default: Go to package first

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    elif event.key == pygame.K_g: selected_action = 0; print("HL Action: Go To Goal")
                    elif event.key == pygame.K_b: selected_action = 1; print("HL Action: Go To Battery")
                    elif event.key == pygame.K_p: selected_action = 2; print("HL Action: Go To Package")

            if not running: break

            obs, reward, terminated, truncated, info = env_package_h.step(selected_action)
            total_reward += reward
            # env_package_h.render() # Render is called inside step

            if terminated or truncated:
                print(f"Package Hierarchy Episode finished! Total reward: {total_reward:.2f}, Info: {info}")
                total_reward = 0
                obs, info = env_package_h.reset() # Reset
                terminated = truncated = False
                selected_action = 2 # Reset action to go for package

        env_package_h.close()
        print("Package Hierarchy Test Done.")

    except (FileNotFoundError, RuntimeError, ValueError, Exception) as e:
        print(f"\nERROR setting up/running Package Hierarchy Test: {e}")
        print("Please ensure policy paths and env IDs are correct.")