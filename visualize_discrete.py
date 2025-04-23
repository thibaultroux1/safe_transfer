import sys
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pandas as pd
import tyro

import safe_transfer.environments
from safe_transfer.nn import AgentDiscrete as Agent
from safe_transfer.parser import parse_env_kwargs


@dataclass
class Args:
    """Script arguments"""
    env_id: str                      # ID of the Gymnasium environment
    model_path: str                  # Path to the trained model checkpoint
    num_episodes: int = 5            # Number of episodes to run
    eval: bool = False               # If True, run evaluation mode (no rendering, saves results)


# ------------------------------
# Visualization and Evaluation
# ------------------------------

def visualize_agent(args: Args):
    """
    Visualize the agent's performance with rendering.
    Loads a single model specified by model_path.
    """
    print("Starting visualization mode... 👀")

    # Create environment with rendering enabled
    # Pass environment-specific arguments parsed earlier
    env = gym.make(args.env_id, render_mode="human", **args.env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env) # Track episode returns and lengths

    # Create a dummy single environment to get observation/action space info for the agent
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(args.env_id, **args.env_kwargs)])
    agent = Agent(envs) # Initialize the standard feed-forward agent
    checkpoint = torch.load(args.model_path, map_location='cpu') # Load model weights
    agent.load_state_dict(checkpoint["model_state_dict"], strict=False)
    agent.eval() # Set agent to evaluation mode (e.g., disable dropout)

    # Set seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    print(f"Running {args.num_episodes} episodes for visualization.")
    for episode in range(args.num_episodes):
        obs, _ = env.reset(seed=episode) # Reset environment for a new episode
        done = False
        total_reward = 0.0

        while not done:
            # Convert observation to tensor and add batch dimension
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad(): # Disable gradient calculation for inference
                # Get action from the agent
                action, _, _, _ = agent.get_action_and_value(obs_tensor)

            action_item = action.item() # Get the action as an integer

            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action_item)
            done = terminated or truncated # Episode ends if terminated or truncated
            total_reward += reward


        print(f"Episode {episode + 1} finished with reward {total_reward:.2f} 🎉")
        if "episode" in info:
             print(f"Episode stats: length={info['episode']['l']}, return={info['episode']['r']}")

    env.close()


def evaluate_agent(args: Args):
    """
    Evaluate the agent's performance without rendering.
    Records performance metrics per episode.
    """
    print("Starting evaluation mode... 🧪")

    # Create environment without rendering, potentially with a maximum step limit
    env = gym.make(args.env_id, render_mode=None, **args.env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env) # Track episode stats

    # Initialize the agent (same as in visualization)
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(args.env_id, **args.env_kwargs)])
    agent = Agent(envs)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    results = [] # List to store results for each episode

    print(f"Running {args.num_episodes} episodes for evaluation.")
    # Set seeds for reproducible evaluation across runs
    env.action_space.seed(0)
    env.observation_space.seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    for episode in range(args.num_episodes):
        obs, _ = env.reset(seed=episode) # Use episode number as seed for consistency
        done = False
        total_reward = 0.0
        step_count = 0
        episode_info = {}  # Store all info for this episode

        while not done:
            step_count += 1
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                # Use deterministic actions for evaluation
                action, _, _, _ = agent.get_action_and_value(obs_tensor)

            action_item = action.item()

            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action_item)
            done = terminated or truncated
            total_reward += reward
            
            # Store all info from this step
            for key, value in info.items():
                if key not in episode_info:
                    episode_info[key] = []
                episode_info[key].append(value)

        # Record results for this episode
        episode_result = {
            "seed": episode,
            "steps": step_count,
            "env_params": str(args.env_kwargs),
            "model_path": args.model_path,
            "return": info.get("episode", {}).get("r", total_reward),
            "terminated": terminated,
            "truncated": truncated,
        }
        
        # Add all info from the episode
        for key, values in episode_info.items():
            if isinstance(values, list):
                # For lists, store both the full list and some statistics
                episode_result[f"{key}_all"] = values
                if all(isinstance(v, (int, float)) for v in values):
                    episode_result[f"{key}_mean"] = sum(values) / len(values)
                    episode_result[f"{key}_min"] = min(values)
                    episode_result[f"{key}_max"] = max(values)
            else:
                episode_result[key] = values

        results.append(episode_result)
        print(f"Episode {episode} finished in {step_count} steps | Return: {episode_result['return']}")

    env.close()

    # Calculate and print overall success rate
    if results:
        avg_return = sum(r["return"] for r in results) / len(results)
        avg_steps = sum(r["steps"] for r in results) / len(results)
        print(f"\n--- Evaluation Summary ---")
        print(f"Episodes: {len(results)}")
        print(f"Average Return: {avg_return}")
        print(f"Average Steps: {avg_steps}")

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        output_filename = f"evaluation_results_{args.env_id}.csv"
        print(f"\nSaving results to '{output_filename}' 📄")
        results_df.to_csv(output_filename, index=False)
    else:
        print("No episodes were run.")


if __name__ == "__main__":
    # Parse environment-specific keyword arguments first (for example, --env.n_steps 5000)
    env_kwargs, remaining_argv = parse_env_kwargs(sys.argv[1:])
    # Parse the remaining standard arguments using tyro
    args = tyro.cli(Args, args=remaining_argv)
    # Store the parsed environment kwargs in the args object
    args.env_kwargs = env_kwargs
    print("Environment ID:", args.env_id)
    print("Model Path:", args.model_path)
    print("Num Episodes:", args.num_episodes)
    print("Evaluation Mode:", args.eval)
    print("Environment Kwargs:", args.env_kwargs) # Print parsed env_kwargs for verification

    # Run either evaluation or visualization based on the --eval flag
    if args.eval:
        evaluate_agent(args)
    else:
        visualize_agent(args)
