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

import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium.spaces import Dict, MultiBinary, Box, Discrete

class SimpleKidneyEnv(gym.Env):
    def __init__(self, n_agents=50, p=0.1, q=0.4, pct_hard=0.6, n_timesteps=100, time_low=4, time_high=8):
        self.n_agents = n_agents
        self.pct_hard = pct_hard
        self.p = p  # Hard-Easy match probability
        self.q = q  # Easy-Easy match probability
        self.n_timesteps = n_timesteps
        self.action_space = MultiBinary((n_agents, n_agents))
        self.observation_space = Dict({
            "adjacency": MultiBinary((n_agents, n_agents)),
            "active": MultiBinary(n_agents),
            "matched": MultiBinary(n_agents),
            "is_hard": MultiBinary(n_agents),
            "arrivals": Discrete(n_timesteps),
            "departures": Discrete(n_timesteps),
            "timestep": Discrete(n_timesteps)
        })
        self.time_low = time_low
        self.time_high = time_high
        self.reset()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Create patient population with hard-to-match patients
        self.is_hard = np.zeros(self.n_agents)
        hard_indices = self.np_random.choice(self.n_agents, int(self.n_agents * self.pct_hard), replace=False)
        self.is_hard[hard_indices] = 1
        
        # Generate compatibility matrix
        self.compat = np.zeros((self.n_agents, self.n_agents))
        r = self.np_random.random((self.n_agents, self.n_agents))
        easy_indices = np.where(self.is_hard == 0)[0]
        self.compat[np.ix_(hard_indices, easy_indices)] = r[np.ix_(hard_indices, easy_indices)] < self.p
        self.compat[np.ix_(easy_indices, hard_indices)] = r[np.ix_(easy_indices, hard_indices)] < self.p
        self.compat[np.ix_(easy_indices, easy_indices)] = r[np.ix_(easy_indices, easy_indices)] < self.q
        
        # Generate arrivals (evenly distributed to ensure activity every step)
        arrivals_per_step = max(1, self.n_agents // self.n_timesteps)
        self.arrivals = []
        for t in range(self.n_timesteps):
            if t < self.n_agents // arrivals_per_step:
                self.arrivals.extend([t] * arrivals_per_step)
        remaining = self.n_agents - len(self.arrivals)
        if remaining > 0:
            self.arrivals.extend([0] * remaining)
        self.arrivals = np.array(self.arrivals)
        self.np_random.shuffle(self.arrivals)
        
        # Generate departure times
        self.departures = np.minimum(self.arrivals + self.np_random.integers(self.time_low, self.time_high, self.n_agents), self.n_timesteps)
        
        self.active = np.zeros(self.n_agents)
        self.matched = np.zeros(self.n_agents)
        self.current_step = 0
        self.graph = nx.DiGraph()
        for i in range(self.n_agents):
            self.graph.add_node(i)
            
        return self._get_obs(), {}
        
    def step(self, action, method="max_weight"):
        action = action > 0.5
        prev_matched = np.copy(self.matched)
        
        # Process matchings if there are valid edges
        adj_matrix = nx.adjacency_matrix(self.graph).toarray()

        valid_edges = adj_matrix * np.outer(self.active, self.active) * action
        
        if np.sum(valid_edges) > 0:
            # Create subgraph of selected edges
            edges = np.where(valid_edges > 0)
            undirected = nx.Graph()
            
            # Only keep mutual edges for matching
            for i in range(len(edges[0])):
                u, v = edges[0][i], edges[1][i]
                if valid_edges[v, u] > 0:  # If mutual edge exists
                    undirected.add_edge(u, v)
            
            if method == "max_weight":
                # Use maximum cardinality matching
                matches = nx.max_weight_matching(undirected)
                for u, v in matches:
                    self._match_nodes(u, v)
            else:
                # Use greedy matching prioritizing hard-to-match patients
                self._greedy_matching(undirected)
        
        # Process arrivals and add new edges
        new_arrivals = np.where(self.arrivals == self.current_step)[0]
        for i in new_arrivals:
            self.active[i] = 1
            for j in range(self.n_agents):
                if i != j and self.active[j]:
                    if self.compat[i, j]:
                        self.graph.add_edge(i, j)
                    if self.compat[j, i]:
                        self.graph.add_edge(j, i)
        
        # Process departures
        departures = np.where(self.departures == self.current_step)[0]
        for i in departures:
            if self.active[i]:
                self.active[i] = 0
                self.graph.remove_edges_from(list(self.graph.in_edges(i)) + list(self.graph.out_edges(i)))
        
        # Calculate reward and check termination
        self.current_step += 1
        done = self.current_step >= self.n_timesteps
        
        # Calculate weighted reward based on hard/easy matches
        newly_matched = self.matched - prev_matched
        hard_matched = np.sum(newly_matched * self.is_hard) / max(1, np.sum(self.is_hard))
        easy_matched = np.sum(newly_matched * (1 - self.is_hard)) / max(1, np.sum(1 - self.is_hard))
        reward = hard_matched * 1.5 + easy_matched * 0.5
        
        return self._get_obs(), reward, done, done, {"total_matched": np.sum(self.matched)}
    
    def _match_nodes(self, u, v):
        # Mark nodes as matched and remove their edges
        for node in [u, v]:
            self.graph.remove_edges_from(list(self.graph.in_edges(node)) + list(self.graph.out_edges(node)))
            self.active[node] = 0
            self.matched[node] = 1
    
    def _greedy_matching(self, graph):
        # Sort edges by priority (hard-to-match patients first)
        edges = [(u, v, self.is_hard[u] + self.is_hard[v]) for u, v in graph.edges()]
        edges.sort(key=lambda x: x[2], reverse=True)
        
        matched_nodes = set()
        for u, v, _ in edges:
            if u not in matched_nodes and v not in matched_nodes:
                self._match_nodes(u, v)
                matched_nodes.update([u, v])
    
    def _get_obs(self):
        return {
            "adjacency": nx.adjacency_matrix(self.graph).toarray().astype(np.int8),
            "active": self.active.astype(np.int8),
            "matched": self.matched.astype(np.int8),
            "is_hard": self.is_hard.astype(np.int8),
            "timestep": self.current_step,
            "arrivals": self.arrivals,
            "departures": self.departures,
        }
        
    def get_greedy_percentage(self):
        # Clone environment and run with greedy matching
        env_copy = SimpleKidneyEnv(self.n_agents, self.p, self.q, self.pct_hard, self.n_timesteps)
        env_copy.reset(seed=getattr(self, "seed", None))
        env_copy.compat = self.compat.copy()
        env_copy.arrivals = self.arrivals.copy()
        env_copy.departures = self.departures.copy()
        env_copy.is_hard = self.is_hard.copy()
        
        done = False
        total_reward = 0
        while not done:
            # Create full action matrix (consider all edges)
            action = np.ones((self.n_agents, self.n_agents))
            _, reward, done, _, _ = env_copy.step(action, method="greedy")
            total_reward += reward
            
        return total_reward, np.sum(env_copy.matched) / self.n_agents

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

        output_matrix = torch.zeros((edge_index.shape[0], edge_index.shape[1]), device=edge_index.device)
        for i in range(len(edge_ids)):
            u, v = edge_ids[i]
            output_matrix[u, v] = edges[i]

        return output_matrix

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

def convert_obs_to_tensors(obs, total_timesteps):
    n_agents = obs["active"].shape[0]
    agent_vecs = torch.zeros((n_agents, 16))
    for agent_idx in range(n_agents):
        # Feature population for both active and inactive agents
        if (obs["departures"][agent_idx] - obs["arrivals"][agent_idx]) == 0:
            print("Offending departure: ", obs["departures"][agent_idx])
            print("Offending arrival: ", obs["arrivals"][agent_idx])
            
        agent_vecs[agent_idx, 0] = (obs["timestep"] - obs["arrivals"][agent_idx]) / max(1, obs["departures"][agent_idx] - obs["arrivals"][agent_idx])
        agent_vecs[agent_idx, 1] = obs["is_hard"][agent_idx]
        agent_vecs[agent_idx, 2] = obs["timestep"] / total_timesteps
        agent_vecs[agent_idx, 3] = obs["active"][agent_idx]  # Add active status as feature
        
        # Time-based features
        agent_vecs[agent_idx, 4] = obs["timestep"] % 3
        agent_vecs[agent_idx, 5] = obs["timestep"] % 4
        agent_vecs[agent_idx, 6] = obs["timestep"] % 5
        agent_vecs[agent_idx, 7] = obs["timestep"] % 7
        
        # Add urgency feature for all agents
        agent_vecs[agent_idx, 8] = max(0, (obs["departures"][agent_idx] - obs["timestep"])) / total_timesteps
        agent_vecs[agent_idx, 9] = ((obs["departures"][agent_idx] - obs["timestep"]) <= 2) * 1.0
    return agent_vecs, torch.tensor(obs["adjacency"])


policy = PKEModel(node_input_feature_dim=16)
policy.reset_parameters()
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

n_envs = 128
envs_per_batch = 8
envs_per_test = 4

env = SimpleKidneyEnv(n_timesteps=30, time_low=4, time_high=6)

for env_idx in tqdm(range(n_envs)):
    obs, _ = env.reset()

    done = False
    while not done:
        agent_vecs, edge_index = convert_obs_to_tensors(obs, env.n_timesteps)
        if np.sum(obs["adjacency"]) == 0:
            action = np.zeros((env.n_agents, env.n_agents))
        else:
            action = policy.sample_action(agent_vecs, edge_index).cpu().detach().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    policy.finish_episode(reward)

    if (env_idx + 1) % envs_per_batch == 0:
        finish_minibatch(policy, optimizer)
        policy.eval()
        greedy_rewards = []
        model_rewards = []
        ratios = []

        for _ in tqdm(range(envs_per_test)):
            obs, _ = env.reset()
            done = False
            while not done:
                agent_vecs, edge_index = convert_obs_to_tensors(obs, env.n_timesteps)
                if np.sum(obs["adjacency"]) == 0:
                    action = np.zeros((env.n_agents, env.n_agents))
                else:
                    action = policy(agent_vecs, edge_index).cpu().detach().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            greedy_reward, greedy_percentage = env.get_greedy_percentage()
            model_percentage = np.sum(env.matched) / env.n_agents
            ratio = model_percentage / greedy_percentage
            greedy_rewards.append(greedy_percentage)
            model_rewards.append(model_percentage)
            ratios.append(ratio)

        print(f"After environment {env_idx}:")
        print(f"    Greedy reward: {np.mean(greedy_rewards)}")
        print(f"    Model reward: {np.mean(model_rewards)}")
        print(f"    Ratio: {np.mean(ratios)}")
        print(f"    Std: {np.std(ratios)}")
        print(f"    Min: {np.min(ratios)}")
        print(f"    Max: {np.max(ratios)}")

        policy.train()

