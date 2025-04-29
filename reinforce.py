import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium.spaces import Graph, MultiBinary, Dict, Box, Discrete, MultiDiscrete
from typing import Tuple, Optional
from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GAT, GlobalAttention, GATv2Conv, global_mean_pool
from environment_reinforce import PairedKidneyDonationEnv
from tqdm import tqdm
from torch.distributions import Bernoulli
from torch_geometric.utils import dense_to_sparse


# Create a simple graph-based model that will determine, for each edge, whether or not it is a good time to match at this time or not.
class PKEModel(nn.Module):
    def __init__(self, node_input_feature_dim=16, hidden_dim=64, num_heads=4):
        super(PKEModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.proj1 = nn.Linear(node_input_feature_dim, self.hidden_dim)
        self.proj2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.conv1 = GATv2Conv(self.hidden_dim, self.hidden_dim // self.num_heads, heads=4, dropout=0.1)
        self.proj4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.conv2 = GATv2Conv(self.hidden_dim, self.hidden_dim // self.num_heads, heads=4, dropout=0.1)
        self.proj5 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # now, project the output for each edge
        self.edge_proj1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.edge_proj2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.edge_proj3 = nn.Linear(self.hidden_dim, 1)

        self.episode_logprobs = torch.tensor([])
        self.rewards = []
        self.logprobs = []

    def forward(self, x, edge_index):
        # only made to process a single graph at a time
        x = F.dropout(F.relu(self.proj1(x)), p=0.2)
        x = x + F.dropout(F.relu(self.proj2(x)), p=0.2)

        sparse_edge_index = dense_to_sparse(edge_index)[0]

        x = x + self.conv1(x, sparse_edge_index)
        x = x + F.dropout(F.relu(self.proj4(x)), p=0.2)
        x = x + self.conv2(x, sparse_edge_index)
        x = x + F.dropout(F.relu(self.proj5(x)), p=0.2)

        # now, for each of the edges, create a vector
        edges = []
        edge_ids = []
        for i in range(edge_index.shape[0]):
            for j in range(edge_index.shape[1]):
                if i != j and edge_index[i, j] == 1:
                    edges.append(torch.cat([x[i], x[j]], dim=-1))
                    edge_ids.append((i, j))
        
        edges = torch.stack(edges, dim=0)

        # process the edges
        edges = self.edge_proj1(edges)
        edges = F.dropout(F.relu(self.edge_proj2(edges)), p=0.2)
        edges = F.sigmoid(self.edge_proj3(edges))

        return edges

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def sample_action(self, obs, edge_index):
        probs = self(obs, edge_index)
        dist = Bernoulli(probs=probs)
        action = dist.sample()
        logprobs = dist.log_prob(action)
        if self.episode_logprobs.numel() == 0:
            self.episode_logprobs = logprobs.clone()
        else:
            self.episode_logprobs = torch.cat([self.episode_logprobs, logprobs], dim=0)
        # print("Action: ", action)
        return action
    
    def finish_episode(self, reward):
        # print("Finishing episode with reward: ", reward)
        # print("Logprobs: ", self.logprobs)
        self.logprobs.append(torch.sum(self.episode_logprobs))
        self.rewards.append(reward)
        self.episode_logprobs = torch.tensor([])
        

def finish_minibatch(model, optimizer):
    rewards = torch.tensor(model.rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    logprobs = torch.stack(model.logprobs)
    optimizer.zero_grad()
    loss = -torch.mean(logprobs * rewards)
    loss.backward()
    optimizer.step()

    model.logprobs = []
    model.rewards = []

def convert_obs_to_tensors(obs):
    # observation:
    # - adjacency matrix
    # - current timestep
    # - arrivals
    # - departure
    # - is_hard_to_match
    # - active_agents
    # - matched_agents
    # - total_timesteps
    n_agents = obs["active_agents"].shape[0]
    agent_vecs = torch.zeros((n_agents, 16))
    for agent_idx in range(n_agents):
        if obs["active_agents"][agent_idx] == 0:
            if (obs["departures"][agent_idx] - obs["arrivals"][agent_idx]) == 0:
                print("Offending departure: ", obs["departures"][agent_idx])
                print("Offending arrival: ", obs["arrivals"][agent_idx])
                
            agent_vecs[agent_idx, 0] = (obs["timestep"] - obs["arrivals"][agent_idx]) / (obs["departures"][agent_idx] - obs["arrivals"][agent_idx])
            agent_vecs[agent_idx, 1] = obs["is_hard_to_match"][agent_idx]
            agent_vecs[agent_idx, 2] = obs["timestep"] / obs["total_timesteps"]
            
            agent_vecs[agent_idx, 3] = obs["timestep"] % 3
            agent_vecs[agent_idx, 4] = obs["timestep"] % 4
            agent_vecs[agent_idx, 5] = obs["timestep"] % 5
            agent_vecs[agent_idx, 6] = obs["timestep"] % 7

            agent_vecs[agent_idx, 7] = ((obs["departures"][agent_idx] - obs["timestep"]) <= 2) * 1.0
    return agent_vecs, torch.tensor(obs["adjacency_matrix"])


policy = PKEModel(node_input_feature_dim=16)
policy.reset_parameters()
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

n_envs = 128
envs_per_batch = 8
envs_per_test = 4

env = PairedKidneyDonationEnv(n_agents=50, n_timesteps=16, death_range=[6, 8])

for env_idx in tqdm(range(n_envs)):
    obs, _ = env.reset()

    done = False
    while not done:
        agent_vecs, edge_index = convert_obs_to_tensors(obs)
        action = policy.sample_action(agent_vecs, edge_index)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    policy.finish_episode(reward)

    if (env_idx + 1) % envs_per_batch == 0:
        finish_minibatch(policy, optimizer)
        policy.eval()
        greedy_rewards = []
        model_rewards = []
        theoretical_maxes = []
        ratios = []

        for _ in tqdm(range(envs_per_test)):
            obs, _ = env.reset()
            done = False
            while not done:
                agent_vecs, edge_index = convert_obs_to_tensors(obs)
                action = policy(agent_vecs, edge_index)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            greedy_reward, greedy_percentage = env.get_greedy_percentage()
            model_percentage = np.sum(env.matched_agents) / env.n_agents
            ratio = model_percentage / greedy_percentage
            greedy_rewards.append(greedy_percentage)
            model_rewards.append(model_percentage)
            theoretical_maxes.append(env.get_two_cycle_theoretical_max())
            ratios.append(ratio)

        print(f"After environment {env_idx}:")
        print(f"    Greedy reward: {np.mean(greedy_rewards)}")
        print(f"    Model reward: {np.mean(model_rewards)}")
        print(f"    Theoretical max: {np.mean(theoretical_maxes)}")
        print(f"    Ratio: {np.mean(ratios)}")
        print(f"    Std: {np.std(ratios)}")
        print(f"    Min: {np.min(ratios)}")
        print(f"    Max: {np.max(ratios)}")

        policy.train()
            
