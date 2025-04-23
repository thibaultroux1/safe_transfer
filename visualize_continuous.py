import sys
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pandas as pd
import tyro

import safe_transfer.environments # Register environments
from safe_transfer.nn import AgentContinuous as Agent
from safe_transfer.parser import parse_env_kwargs


@dataclass
class Args:
    """Script arguments"""
    env_id: str                      # ID of the Gymnasium environment (must be continuous action space)
    model_path: str                  # Path to the trained model checkpoint
    num_episodes: int = 5            # Number of episodes to run
    eval: bool = False               # If True, run evaluation mode (no rendering, saves results)


# ------------------------------
# Visualization and Evaluation
# ------------------------------

def visualize_agent(args: Args):
    """
    Visualize the agent's performance with rendering for continuous actions.
    Loads a single model specified by model_path.
    Uses sampled actions from the policy distribution.
    """
    print("Starting visualization mode ... 👀")

    # Create environment with rendering enabled
    env = gym.make(args.env_id, render_mode="human", **args.env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env) # Track episode returns and lengths
    env = gym.wrappers.ClipAction(env)

    # Create a dummy single environment to get observation/action space info for the agent
    # Need this even if rendering the main env, to configure the Agent
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(args.env_id, **args.env_kwargs)])
    agent = Agent(envs) # Initialize agent
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
            with torch.no_grad():
                action_tensor, _, _, _ = agent.get_action_and_value(obs_tensor)

            action_np = action_tensor.squeeze(0).cpu().numpy()

            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated # Episode ends if terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1} finished with reward {total_reward:.2f} 🎉")
        if "episode" in info:
             print(f"Episode stats: length={info['episode']['l']}, return={info['episode']['r']}")

    env.close()
    envs.close() # Close the dummy envs too


def evaluate_agent(args: Args):
    """
    Evaluate the agent's performance without rendering.
    Records performance metrics per episode.
    Uses deterministic actions (mean of the distribution).
    """
    print("Starting evaluation mode ... 🧪")

    # Create environment without rendering
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
                action_mean_tensor = agent.actor_mean(obs_tensor)

            action_np = action_mean_tensor.squeeze(0).cpu().numpy()

            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action_np)
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
            # Use recorded episode return if available, otherwise use manually tracked total_reward
            "return": info.get("episode", {}).get("r", total_reward),
            "terminated": terminated,
            "truncated": truncated,
        }

        # Add all info from the episode (aggregate if possible)
        for key, values in episode_info.items():
            # Avoid adding the 'episode' dict itself again
            if key == "episode":
                continue
            if isinstance(values, list):
                # For lists, store both the full list and some statistics if numerical
                episode_result[f"{key}_all"] = values
                # Check if all elements are numbers before calculating stats
                if values and all(isinstance(v, (int, float, np.number)) for v in values):
                    try:
                         episode_result[f"{key}_mean"] = np.mean(values)
                         episode_result[f"{key}_min"] = np.min(values)
                         episode_result[f"{key}_max"] = np.max(values)
                         episode_result[f"{key}_sum"] = np.sum(values) # Sum might be relevant too
                    except (TypeError, ValueError):
                        # Handle cases where stats can't be computed (e.g., mixed types unexpectedly)
                        print(f"Warning: Could not compute stats for key '{key}' in episode {episode}")

            else:
                episode_result[key] = values # Store non-list info directly


        results.append(episode_result)
        print(f"Episode {episode+1} finished in {step_count} steps | Return: {episode_result['return']:.2f} (Term: {terminated}, Trunc: {truncated})")

    env.close()
    envs.close() # Close the dummy envs too

    # Calculate and print overall summary statistics
    if results:
        avg_return = np.mean([r["return"] for r in results])
        std_return = np.std([r["return"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])
        num_terminated = sum(r["terminated"] for r in results)
        num_truncated = sum(r["truncated"] for r in results)

        print(f"\n--- Evaluation Summary ---")
        print(f"Episodes Run: {len(results)}")
        print(f"Avg Return:   {avg_return:.2f} +/- {std_return:.2f}")
        print(f"Avg Steps:    {avg_steps:.2f}")
        print(f"Terminated:   {num_terminated}/{len(results)}")
        print(f"Truncated:    {num_truncated}/{len(results)}")

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        # Create a filename that is less likely to overwrite other results
        env_name_part = args.env_id.replace('/','_') # Handle potential slashes in env_id
        model_name_part = args.model_path.split('/')[-1].replace('.pt','') # Get model filename part
        output_filename = f"evaluation_results_{env_name_part}_{model_name_part}_ep{args.num_episodes}.csv"
        print(f"\nSaving detailed results to '{output_filename}' 📄")
        try:
            results_df.to_csv(output_filename, index=False)
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
            # Fallback: print dataframe to console if saving fails
            print("\nResults DataFrame:")
            print(results_df)

    else:
        print("No episodes were run.")


if __name__ == "__main__":
    # Parse environment-specific keyword arguments first (e.g., --env.n_steps 5000)
    env_kwargs, remaining_argv = parse_env_kwargs(sys.argv[1:])
    # Parse the remaining standard arguments using tyro
    args = tyro.cli(Args, args=remaining_argv)
    # Store the parsed environment kwargs in the args object
    args.env_kwargs = env_kwargs
    print("--- Script Configuration ---")
    print(f"Environment ID:   {args.env_id}")
    print(f"Model Path:       {args.model_path}")
    print(f"Num Episodes:     {args.num_episodes}")
    print(f"Evaluation Mode:  {args.eval}")
    print(f"Environment Kwargs: {args.env_kwargs}") # Print parsed env_kwargs for verification
    print("-" * 26)


    # Run either evaluation or visualization based on the --eval flag
    if args.eval:
        evaluate_agent(args)
    else:
        visualize_agent(args)