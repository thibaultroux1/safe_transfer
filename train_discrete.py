import os
import sys
import random
import time
from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import safe_transfer.environments
from safe_transfer.nn import layer_init
from safe_transfer.parser import parse_env_kwargs
from safe_transfer.nn import AgentDiscrete as Agent
from safe_transfer.ewc import consolidate_params, estimate_fisher, ewc_penalty


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = None
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    env_id: str = "Drone2D-v0"
    """the id of the environment"""
    pretrained_env_id: str = None
    """the id of the environment the pretrained model was trained on. Defaults to env_id if None"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    pretrained_model_path: str = None # Path to the pretrained model
    """path to pretrained model to load and extend"""
    guided_exploration: bool = False
    """guided exploration by the pretrained model"""
    initial_guide_weight: float = 0.1
    """weight use for the guidance"""
    bypass_action: bool = False
    """Source policy bypasses the action when the action is 'Go to battery' """

    # EWC specific arguments
    ewc: bool = False
    """whether to use Elastic Weight Consolidation"""
    ewc_lambda: float = 0.1
    """weight of the EWC penalty term"""
    ewc_sample_size: int = 10000
    """number of samples to use for Fisher information matrix estimation"""
    ewc_batch_size: int = 4096
    """batch size for Fisher information matrix estimation"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    episode_log_interval: int = 50
    """Number of episodes to accumulate before logging episode statistics."""


def make_env(args):
    def thunk():
        env = gym.make(args.env_id, **args.env_kwargs)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
        return env
    return thunk

if __name__ == "__main__":
    args = Args()
    env_kwargs, remaining_argv = parse_env_kwargs(sys.argv[1:])
    args.env_kwargs = env_kwargs
    sys.argv = [sys.argv[0]] + remaining_argv
    args = tyro.cli(Args)
    args.env_kwargs = env_kwargs # Re-assign env_kwargs to args.env_kwargs to ensure they are not overwritten by tyro if accidentally included in standard args
    print("Env Kwargs:", args.env_kwargs) # Print parsed env_kwargs for verification

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}/{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args) for _ in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    pretrained_agent = None
    if args.pretrained_model_path:
        pretrained_env_id_to_use = args.pretrained_env_id if args.pretrained_env_id else args.env_id # Use pretrained_env_id if provided, else default to current env_id
        args_pretrained = replace(args, env_id=args.pretrained_env_id)
        args_pretrained.env_kwargs = args.env_kwargs.copy()
        args_pretrained.env_kwargs['low_level_policies_paths'].pop()
        args_pretrained.env_kwargs['env_ids'].pop()
        if 'reward_package' in list(args_pretrained.env_kwargs.keys()):
            del args_pretrained.env_kwargs['reward_package']
        pretrained_agent_env = gym.vector.SyncVectorEnv([make_env(args_pretrained)]) # Dummy env for pretrained agent
        pretrained_agent = Agent(pretrained_agent_env).to(device) # Create dummy agent to load weights
        pretrained_agent.load_state_dict(torch.load(args.pretrained_model_path, map_location=device)['model_state_dict'])
        pretrained_agent.eval()
        print(f"Loaded pretrained model from: {args.pretrained_model_path} trained on env_id: {pretrained_env_id_to_use}")

    # Initialize Main Agent
    agent = Agent(envs, pretrained_agent).to(device)

    # EWC Specific Setup
    if args.ewc:
        if not args.pretrained_model_path:
            print("Warning: EWC flag is set, but no pretrained model path provided. EWC will be disabled.")
            args.ewc = False
        else:
            print("EWC enabled. Estimating Fisher information matrix on initialized agent...")
            start_ewc_setup_time = time.time()

            # Collect data using the initialized agent
            ewc_obs_list = []
            ewc_action_list = []
            temp_envs = gym.vector.SyncVectorEnv([make_env(args) for _ in range(args.num_envs)])
            temp_next_obs, _ = temp_envs.reset(seed=args.seed + 1)
            temp_next_obs = torch.Tensor(temp_next_obs).to(device)
            collected_samples = 0

            agent.eval()
            with torch.no_grad():
                while collected_samples < args.ewc_sample_size:
                    # Get action from the main agent (final structure, pretrained weights)
                    action, _, _, _ = agent.get_action_and_value(temp_next_obs)
                    ewc_obs_list.append(temp_next_obs.cpu())
                    ewc_action_list.append(action.cpu())

                    # Step temporary environment
                    temp_next_obs_np, _, temp_term, temp_trunc, _ = temp_envs.step(action.cpu().numpy())
                    temp_done = np.logical_or(temp_term, temp_trunc)
                    collected_samples += args.num_envs

                    # Reset only the environments that are done
                    done_indices = np.where(temp_done)[0]
                    if done_indices.size > 0:
                        reset_observations, _ = temp_envs.reset(
                            seed=args.seed + collected_samples,
                            options={'indices': done_indices}
                        )
                        temp_next_obs_np[done_indices] = reset_observations[done_indices]
                    temp_next_obs = torch.Tensor(temp_next_obs_np).to(device)

            temp_envs.close()
            agent.train()

            # Create dataset and dataloader
            ewc_obs_tensor = torch.cat(ewc_obs_list, dim=0)[:args.ewc_sample_size]
            ewc_action_tensor = torch.cat(ewc_action_list, dim=0)[:args.ewc_sample_size]
            ewc_dataset = TensorDataset(ewc_obs_tensor, ewc_action_tensor)
            ewc_data_loader = DataLoader(ewc_dataset, batch_size=args.ewc_batch_size, drop_last=(len(ewc_dataset) % args.ewc_batch_size == 0))

            # Estimate Fisher matrix
            fisher_dict = estimate_fisher(agent, ewc_data_loader, args.ewc_sample_size, args.ewc_batch_size, device)

            # Consolidate parameters
            consolidate_params(agent, fisher_dict)
            print(f"Fisher estimation and consolidation complete. Time: {time.time() - start_ewc_setup_time:.2f}s")

    # Optimizer Setup
    agent.train()
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Create checkpoint directory
    checkpoint_dir = f"runs/{run_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_interval = args.total_timesteps // 10  # Save every 10% of total steps
    next_checkpoint_step = checkpoint_interval

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Add episode logging accumulators
    episode_returns = []
    episode_lengths = []
    episode_stats = {k: [] for k in infos.keys() if k[0] != '_'}

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if args.guided_exploration:
            frac = 1.0 - (iteration - 1.0) / (args.num_iterations // 2)
            frac = max(0, frac)
            guide_weight = frac * args.initial_guide_weight

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if not args.guided_exploration:
                    action, logprob, _, value, categorical_target = agent.get_action_and_value(next_obs, return_categorical=True)
                    if args.bypass_action:
                        action_source, _, _, _ = pretrained_agent.get_action_and_value(next_obs[:, :-3])
                        action_go_to_battery_index = 1
                        battery_action_tensor = torch.tensor(action_go_to_battery_index, device=device)
                        override_mask = (action_source == action_go_to_battery_index)
                        action = torch.where(override_mask, battery_action_tensor, action)
                        logprob = categorical_target.log_prob(action)
                else:
                    _, _, _,value, categorical_target = agent.get_action_and_value(next_obs, return_categorical=True)
                    _, _, _, _, categorical_source = pretrained_agent.get_action_and_value(next_obs[:, :-3], return_categorical=True)
                    action_probs_target = categorical_target.probs
                    action_probs_source = categorical_source.probs.cpu()
                    action_probs_source = F.pad(action_probs_source, (0, action_probs_target.shape[1] - action_probs_source.shape[1]))
                    probs_blended = (1 - guide_weight) * action_probs_target + guide_weight * action_probs_source
                    categorical = Categorical(probs_blended)
                    action = categorical.sample()
                    logprob = categorical.log_prob(action)

                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # Determine done status using termination and truncation flags
            current_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Loop over each environment index
            for idx in range(args.num_envs):
                # When an episode is done and hasn't been logged yet:
                if current_done[idx]:
                    info = None
                    if "final_info" in infos and idx < len(infos["final_info"]):
                        info = infos["final_info"][idx]
                    elif idx < len(infos):  # fallback if infos is a list of dicts
                        info = infos[idx]

                    if info and "episode" in info:
                        episode_returns.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
                        for k in episode_stats.keys():
                            episode_stats[k].append(info.get(k, 0))
                            if len(episode_returns) >= args.episode_log_interval:
                                avg_return = np.mean(episode_returns)
                                avg_length = np.mean(episode_lengths)
                                writer.add_scalar("charts/episodic_return", avg_return, global_step)
                                writer.add_scalar("charts/episodic_length", avg_length, global_step)
                                episode_returns.clear()
                                episode_lengths.clear()
                                for k in episode_stats.keys():
                                    avg_stat = np.mean(episode_stats[k])
                                    writer.add_scalar(f"charts/{k}", avg_stat, global_step)
                                    episode_stats[k].clear()

            # Save checkpoint if we've reached the interval
            if global_step >= next_checkpoint_step:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pt")
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                next_checkpoint_step += checkpoint_interval
                print(f"Saved checkpoint at step {global_step}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        ewc_losses = [] # Reset for each iteration's logging

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                # Calculate EWC penalty
                ewc_loss_term = torch.tensor(0.0).to(device)
                if args.ewc:
                    ewc_loss_term = ewc_penalty(agent, args.ewc_lambda, device)
                    ewc_losses.append(ewc_loss_term.item())

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + ewc_loss_term

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if args.guided_exploration:
            writer.add_scalar("charts/guide_weight", guide_weight, global_step)
        if args.ewc and ewc_losses:
            writer.add_scalar("losses/ewc_penalty", np.mean(ewc_losses), global_step)

    envs.close()
    writer.close()

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pt")
    torch.save({
        'global_step': global_step,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
