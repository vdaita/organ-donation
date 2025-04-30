import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium.spaces import Graph, MultiBinary, Dict, Box, Discrete, MultiDiscrete
from typing import Tuple, Optional
from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GAT, GlobalAttention, GATv2Conv, global_mean_pool
from tqdm import tqdm
from torch.distributions import Bernoulli
from torch_geometric.utils import dense_to_sparse
from torch.optim.lr_scheduler import StepLR

class SimpleKidneyEnv(gym.Env):
    def __init__(self, n_agents=50, p=0.1, q=0.4, pct_hard=0.6, n_timesteps=100, time_low=4, time_high=8):
        self.n_agents = n_agents
        self.pct_hard = pct_hard
        self.p = p
        self.q = q
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
        self.is_hard = np.zeros(self.n_agents)
        hard_indices = self.np_random.choice(self.n_agents, int(self.n_agents * self.pct_hard), replace=False)
        self.is_hard[hard_indices] = 1
        
        self.compat = np.zeros((self.n_agents, self.n_agents))
        r = self.np_random.random((self.n_agents, self.n_agents))
        easy_indices = np.where(self.is_hard == 0)[0]
        self.compat[np.ix_(hard_indices, easy_indices)] = r[np.ix_(hard_indices, easy_indices)] < self.p
        self.compat[np.ix_(easy_indices, hard_indices)] = r[np.ix_(easy_indices, hard_indices)] < self.p
        self.compat[np.ix_(easy_indices, easy_indices)] = r[np.ix_(easy_indices, easy_indices)] < self.q
        
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
        
        adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        valid_edges = adj_matrix * np.outer(self.active, self.active) * action
        
        if np.sum(valid_edges) > 0:
            edges = np.where(valid_edges > 0)
            undirected = nx.Graph()
            
            for i in range(len(edges[0])):
                u, v = edges[0][i], edges[1][i]
                if valid_edges[v, u] > 0:
                    undirected.add_edge(u, v)
            
            if method == "max_weight":
                matches = nx.max_weight_matching(undirected)
                for u, v in matches:
                    self._match_nodes(u, v)
            else:
                self._greedy_matching(undirected)
        
        new_arrivals = np.where(self.arrivals == self.current_step)[0]
        for i in new_arrivals:
            self.active[i] = 1
            for j in range(self.n_agents):
                if i != j and self.active[j]:
                    if self.compat[i, j]:
                        self.graph.add_edge(i, j)
                    if self.compat[j, i]:
                        self.graph.add_edge(j, i)
        
        departures = np.where(self.departures == self.current_step)[0]
        for i in departures:
            if self.active[i]:
                self.active[i] = 0
                self.graph.remove_edges_from(list(self.graph.in_edges(i)) + list(self.graph.out_edges(i)))
        
        self.current_step += 1
        done = self.current_step >= self.n_timesteps
        
        newly_matched = self.matched - prev_matched
        hard_matched = np.sum(newly_matched * self.is_hard) / max(1, np.sum(self.is_hard))
        easy_matched = np.sum(newly_matched * (1 - self.is_hard)) / max(1, np.sum(1 - self.is_hard))
        reward = hard_matched * 1.5 + easy_matched * 0.5
        
        return self._get_obs(), reward, done, done, {"total_matched": np.sum(self.matched)}
    
    def _match_nodes(self, u, v):
        for node in [u, v]:
            self.graph.remove_edges_from(list(self.graph.in_edges(node)) + list(self.graph.out_edges(node)))
            self.active[node] = 0
            self.matched[node] = 1
    
    def _greedy_matching(self, graph):
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
        env_copy = SimpleKidneyEnv(self.n_agents, self.p, self.q, self.pct_hard, self.n_timesteps, self.time_low, self.time_high)
        env_copy.reset(seed=getattr(self, "seed", None))
        env_copy.compat = self.compat.copy()
        env_copy.arrivals = self.arrivals.copy()
        env_copy.departures = self.departures.copy()
        env_copy.is_hard = self.is_hard.copy()
        
        done = False
        total_reward = 0
        while not done:
            action = np.ones((self.n_agents, self.n_agents))
            _, reward, done, _, _ = env_copy.step(action, method="greedy")
            total_reward += reward
            
        return total_reward, np.sum(env_copy.matched) / self.n_agents

class PKEModel(nn.Module):
    def __init__(self, node_input_feature_dim=12, hidden_dim=64, num_heads=4):
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

        self.edge_proj1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.edge_proj2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.edge_proj3 = nn.Linear(self.hidden_dim, 1)

        self.episode_logprobs = []
        self.episode_distributions = []
        self.rewards = []
        self.logprobs = []
        self.entropies = []

    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.proj1(x)), p=0.2)
        x = x + F.dropout(F.relu(self.proj2(x)), p=0.2)

        sparse_edge_index = dense_to_sparse(edge_index)[0]

        x = x + self.conv1(x, sparse_edge_index)
        x = x + F.dropout(F.relu(self.proj4(x)), p=0.2)
        x = x + self.conv2(x, sparse_edge_index)
        x = x + F.dropout(F.relu(self.proj5(x)), p=0.2)

        edges = []
        edge_ids = []
        for i in range(edge_index.shape[0]):
            for j in range(edge_index.shape[1]):
                if i != j and edge_index[i, j] == 1:
                    edges.append(torch.cat([x[i], x[j]], dim=-1))
                    edge_ids.append((i, j))
        
        if not edge_ids:
            return torch.zeros((edge_index.shape[0], edge_index.shape[1]), device=edge_index.device)

        edges = torch.stack(edges, dim=0)

        edges = self.edge_proj1(edges)
        edges = F.dropout(F.relu(self.edge_proj2(edges)), p=0.2)
        edges = torch.sigmoid(self.edge_proj3(edges))

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
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        dist = Bernoulli(probs=probs)
        action = dist.sample()
        self.episode_logprobs.append(dist.log_prob(action))
        self.episode_distributions.append(dist)
        return action
    
    def finish_episode(self, reward):
        if not self.episode_logprobs:
            self.episode_logprobs = []
            self.episode_distributions = []
            return

        episode_total_logprob = torch.sum(torch.stack(self.episode_logprobs))
        episode_total_entropy = torch.sum(torch.stack([d.entropy().sum() for d in self.episode_distributions]))

        self.logprobs.append(episode_total_logprob)
        self.entropies.append(episode_total_entropy)
        self.rewards.append(reward)

        self.episode_logprobs = []
        self.episode_distributions = []

def finish_minibatch(model, optimizer, entropy_coeff=0.01, clip_grad_norm=1.0):
    if not model.rewards:
        print("Warning: Trying to finish minibatch with no episode data.")
        return
        
    rewards = torch.tensor(model.rewards, dtype=torch.float32)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    logprobs = torch.stack(model.logprobs)
    entropies = torch.stack(model.entropies)
    
    policy_loss = -torch.mean(logprobs * rewards)
    entropy_loss = -torch.mean(entropies)
    
    loss = policy_loss + entropy_coeff * entropy_loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    optimizer.step()

    model.logprobs = []
    model.rewards = []
    model.entropies = []

def convert_obs_to_tensors(obs, total_timesteps):
    n_agents = obs["active"].shape[0]
    agent_vecs = torch.zeros((n_agents, 12))
    adj_matrix = obs["adjacency"]
    
    in_degree = np.sum(adj_matrix, axis=0)
    out_degree = np.sum(adj_matrix, axis=1)
    
    max_degree = n_agents - 1

    for agent_idx in range(n_agents):
        time_in_pool = obs["timestep"] - obs["arrivals"][agent_idx]
        total_possible_time = obs["departures"][agent_idx] - obs["arrivals"][agent_idx]
        
        if total_possible_time <= 0:
            time_frac = 1.0
        else:
            time_frac = time_in_pool / total_possible_time
             
        agent_vecs[agent_idx, 0] = time_frac
        agent_vecs[agent_idx, 1] = obs["is_hard"][agent_idx]
        agent_vecs[agent_idx, 2] = obs["timestep"] / total_timesteps
        agent_vecs[agent_idx, 3] = obs["active"][agent_idx]
        
        agent_vecs[agent_idx, 4] = (obs["timestep"] % 3) / 2.0
        agent_vecs[agent_idx, 5] = (obs["timestep"] % 4) / 3.0
        agent_vecs[agent_idx, 6] = (obs["timestep"] % 5) / 4.0
        agent_vecs[agent_idx, 7] = (obs["timestep"] % 7) / 6.0
        
        time_to_departure = obs["departures"][agent_idx] - obs["timestep"]
        agent_vecs[agent_idx, 8] = max(0, time_to_departure) / total_timesteps
        agent_vecs[agent_idx, 9] = (time_to_departure <= 2 and time_to_departure > 0) * 1.0

        agent_vecs[agent_idx, 10] = in_degree[agent_idx] / max(1, max_degree)
        agent_vecs[agent_idx, 11] = out_degree[agent_idx] / max(1, max_degree)

    return agent_vecs, torch.tensor(adj_matrix, dtype=torch.float32)


if __name__ == "__main__":
    policy = PKEModel(node_input_feature_dim=12)
    policy.reset_parameters()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    n_envs = 16384
    envs_per_batch = 32
    envs_per_test = 16

    env = SimpleKidneyEnv(n_agents=100, n_timesteps=32, time_low=8, time_high=12, p=0.037, q=0.087, pct_hard=0.6)

    all_greedy_rewards = []
    all_model_rewards = []
    all_ratios = []

    num_batches = n_envs // envs_per_batch
    for batch_idx in tqdm(range(num_batches), desc="Training Batches"):
        policy.train()
        for env_run in range(envs_per_batch):
            obs, _ = env.reset()
            done = False
            while not done:
                agent_vecs, edge_index = convert_obs_to_tensors(obs, env.n_timesteps)
                
                if torch.sum(edge_index) == 0:
                    action_tensor = torch.zeros_like(edge_index)
                    action = action_tensor.cpu().numpy()
                else:
                    action_tensor = policy.sample_action(agent_vecs, edge_index)
                    action = action_tensor.cpu().detach().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            policy.finish_episode(reward)

        finish_minibatch(policy, optimizer, entropy_coeff=0.01, clip_grad_norm=1.0)
        scheduler.step()

        if (batch_idx + 1) % 5 == 0:
            policy.eval()
            batch_greedy_rewards = []
            batch_model_rewards = []
            batch_ratios = []

            with torch.no_grad():
                for _ in range(envs_per_test):
                    obs, _ = env.reset()
                    initial_compat = env.compat.copy()
                    initial_arrivals = env.arrivals.copy()
                    initial_departures = env.departures.copy()
                    initial_is_hard = env.is_hard.copy()
                    
                    done = False
                    model_total_reward = 0
                    while not done:
                        agent_vecs, edge_index = convert_obs_to_tensors(obs, env.n_timesteps)
                        if torch.sum(edge_index) == 0:
                            action = np.zeros((env.n_agents, env.n_agents))
                        else:
                            action_probs = policy(agent_vecs, edge_index)
                            action = (action_probs > 0.5).cpu().numpy()
                            
                        obs, reward, terminated, truncated, info = env.step(action)
                        model_total_reward += reward
                        done = terminated or truncated
                    
                    env_copy = SimpleKidneyEnv(env.n_agents, env.p, env.q, env.pct_hard, env.n_timesteps, env.time_low, env.time_high)
                    env_copy.reset(seed=getattr(env, "_np_random_seed", None))
                    env_copy.compat = initial_compat
                    env_copy.arrivals = initial_arrivals
                    env_copy.departures = initial_departures
                    env_copy.is_hard = initial_is_hard
                    env_copy.active = np.zeros(env_copy.n_agents)
                    env_copy.matched = np.zeros(env_copy.n_agents)
                    env_copy.current_step = 0
                    env_copy.graph = nx.DiGraph()
                    for i in range(env_copy.n_agents): env_copy.graph.add_node(i)
                    new_arrivals = np.where(env_copy.arrivals == env_copy.current_step)[0]
                    for i in new_arrivals: env_copy.active[i] = 1

                    greedy_done = False
                    greedy_total_reward = 0
                    while not greedy_done:
                        greedy_action = np.ones((env.n_agents, env.n_agents))
                        _, greedy_step_reward, greedy_done, _, _ = env_copy.step(greedy_action, method="greedy")
                        greedy_total_reward += greedy_step_reward

                    greedy_percentage = np.sum(env_copy.matched) / env.n_agents
                    model_percentage = np.sum(env.matched) / env.n_agents
                    
                    ratio = model_percentage / greedy_percentage if greedy_percentage > 0 else 0
                    
                    batch_greedy_rewards.append(greedy_percentage)
                    batch_model_rewards.append(model_percentage)
                    batch_ratios.append(ratio)

            avg_greedy = np.mean(batch_greedy_rewards)
            avg_model = np.mean(batch_model_rewards)
            avg_ratio = np.mean(batch_ratios)
            std_ratio = np.std(batch_ratios)
            min_ratio = np.min(batch_ratios)
            max_ratio = np.max(batch_ratios)

            all_greedy_rewards.append(avg_greedy)
            all_model_rewards.append(avg_model)
            all_ratios.append(avg_ratio)

            print(f"\n--- Evaluation after Batch {batch_idx + 1} ---")
            print(f"    LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"    Avg Greedy Matched %: {avg_greedy:.4f}")
            print(f"    Avg Model Matched %:  {avg_model:.4f}")
            print(f"    Avg Ratio (Model/Greedy): {avg_ratio:.4f}")
            print(f"    Std Ratio: {std_ratio:.4f}")
            print(f"    Min Ratio: {min_ratio:.4f}")
            print(f"    Max Ratio: {max_ratio:.4f}")
            print(f"------------------------------------\n")

