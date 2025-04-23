import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np
import torch

def exponential_moving_average(model1, model2, tau=0.1):
    """
    Performs exponential moving average of the weights of two models.
    
    Args:
    model1 (torch.nn.Module): The model whose weights will be updated.
    model2 (torch.nn.Module): The model with the weights to average.
    tau (float): The averaging factor, between 0 and 1.
    """
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        param1.data.copy_(tau * param2.data + (1 - tau) * param1.data)


def init_layer(layer: nn.Linear, std=np.sqrt(2), bias=0.):
    assert isinstance(layer, nn.Linear)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentContinuous(nn.Module):
    def __init__(self, envs, pretrained_agent=None):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = np.prod(envs.single_action_space.shape)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_shape), std=0.01),
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))
        
        if pretrained_agent is not None:
            self.load_pretrained_weights(pretrained_agent, obs_shape, action_shape)

    def load_pretrained_weights(self, pretrained_agent, current_obs_shape, current_action_shape):
        pretrained_obs_shape = pretrained_agent.actor_mean[0].in_features
        pretrained_action_shape = pretrained_agent.actor_mean[-1].out_features

        old_critic = pretrained_agent.critic
        old_actor_mean = pretrained_agent.actor_mean

        if current_obs_shape > pretrained_obs_shape:
            print(f"Extending input layer from {pretrained_obs_shape} to {current_obs_shape} observations.")
            old_actor_layer = old_actor_mean[0]
            new_actor_layer = nn.Linear(current_obs_shape, 64)
            new_actor_layer.weight.data[:, :old_actor_layer.in_features].copy_(old_actor_layer.weight.data)
            new_actor_layer.bias.data.copy_(old_actor_layer.bias.data)
            torch.nn.init.zeros_(new_actor_layer.weight.data[:, old_actor_layer.in_features:])
            old_actor_mean[0] = new_actor_layer

            old_critic_layer = old_critic[0]
            new_critic_layer = nn.Linear(current_obs_shape, 64)
            new_critic_layer.weight.data[:, :old_critic_layer.in_features].copy_(old_critic_layer.weight.data)
            new_critic_layer.bias.data.copy_(old_critic_layer.bias.data)
            torch.nn.init.zeros_(new_critic_layer.weight.data[:, old_critic_layer.in_features:])
            old_critic[0] = new_critic_layer

        if current_action_shape > pretrained_action_shape:
            print(f"Extending output layer from {pretrained_action_shape} to {current_action_shape} actions.")
            old_actor_output_layer = old_actor_mean[-1]
            new_actor_output_layer = nn.Linear(64, current_action_shape)
            new_actor_output_layer.weight.data[:old_actor_output_layer.out_features, :].copy_(old_actor_output_layer.weight.data)
            new_actor_output_layer.bias.data[:old_actor_output_layer.out_features].copy_(old_actor_output_layer.bias.data)
            torch.nn.init.zeros_(new_actor_output_layer.weight.data[old_actor_output_layer.out_features:, :])
            new_actor_output_layer.bias.data[old_actor_output_layer.out_features:].zero_()
            old_actor_mean[-1] = new_actor_output_layer
            self.actor_logstd = nn.Parameter(torch.zeros(1, current_action_shape))

        self.critic = old_critic
        self.actor_mean = old_actor_mean

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class AgentDiscrete(nn.Module):
    def __init__(self, envs, pretrained_agent=None):
        super().__init__()

        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = envs.single_action_space.n
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

        if pretrained_agent is not None:
            self.load_pretrained_weights(pretrained_agent, obs_shape, action_shape)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, return_categorical=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        if not return_categorical:
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)
        else:
            return action, probs.log_prob(action), probs.entropy(), self.critic(x), probs

    def load_pretrained_weights(self, pretrained_agent, current_obs_shape, current_action_shape):
        pretrained_obs_shape = pretrained_agent.actor[0].in_features
        pretrained_action_shape = pretrained_agent.actor[-1].out_features

        old_critic = nn.Sequential(*[layer for layer in pretrained_agent.critic.children()]) # Deep copy of Sequential
        old_actor = nn.Sequential(*[layer for layer in pretrained_agent.actor.children()])   # Deep copy of Sequential

        if current_obs_shape > pretrained_obs_shape:
            print(f"Extending input layer from {pretrained_obs_shape} to {current_obs_shape} observations.")
            old_actor_layer = old_actor[0]
            new_actor_layer = nn.Linear(current_obs_shape, 64)
            new_actor_layer.weight.data[:, :old_actor_layer.in_features].copy_(old_actor_layer.weight.data)
            new_actor_layer.bias.data.copy_(old_actor_layer.bias.data)
            torch.nn.init.zeros_(new_actor_layer.weight.data[:, old_actor_layer.in_features:])
            old_actor[0] = new_actor_layer

            old_critic_layer = old_critic[0]
            new_critic_layer = nn.Linear(current_obs_shape, 64)
            new_critic_layer.weight.data[:, :old_critic_layer.in_features].copy_(old_critic_layer.weight.data)
            new_critic_layer.bias.data.copy_(old_critic_layer.bias.data)
            torch.nn.init.zeros_(new_critic_layer.weight.data[:, old_critic_layer.in_features:])
            old_critic[0] = new_critic_layer

        if current_action_shape > pretrained_action_shape:
            print(f"Extending output layer from {pretrained_action_shape} to {current_action_shape} actions.")
            old_actor_output_layer = old_actor[-1]
            new_actor_output_layer = nn.Linear(64, current_action_shape)
            new_actor_output_layer.weight.data[:old_actor_output_layer.out_features, :].copy_(old_actor_output_layer.weight.data)
            new_actor_output_layer.bias.data[:old_actor_output_layer.out_features].copy_(old_actor_output_layer.bias.data)
            torch.nn.init.zeros_(new_actor_output_layer.weight.data[old_actor_output_layer.out_features:, :])
            new_actor_output_layer.bias.data[old_actor_output_layer.out_features:].fill_(-10.0)
            old_actor[-1] = new_actor_output_layer

        self.critic = old_critic
        self.actor = old_actor

    def freeze_weights(self, freeze_actor=True, freeze_critic=True, neuron_index=10):
        """Freeze specific weights in the actor and critic networks."""
        if freeze_actor:
            actor_layer = self.actor[0]  # Input layer
            with torch.no_grad():
                actor_layer.weight[neuron_index] = actor_layer.weight[neuron_index].detach()
                actor_layer.weight[neuron_index].requires_grad = False
            actor_layer.bias.requires_grad = False  # Freeze bias as well

        if freeze_critic:
            critic_layer = self.critic[0]  # Input layer
            with torch.no_grad():
                critic_layer.weight[neuron_index] = critic_layer.weight[neuron_index].detach()
                critic_layer.weight[neuron_index].requires_grad = False
            critic_layer.bias.requires_grad = False  # Freeze bias as well