# Start off with a random decision-maker
import torch
from torch import nn
from environment import PairedKidneyDonationEnv
import numpy as np
from reinforce import convert_obs_to_tensors
from model import PairedKidneyModel
from typing import List
from tqdm import tqdm
import torch.multiprocessing as mp
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
import gymnasium as gym
import copy
import concurrent.futures

class SimpleBatchedPKEModel(nn.Module):
    def __init__(self, node_input_feature_dim=12, hidden_dim=32, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.node_proj = nn.Linear(node_input_feature_dim, hidden_dim)
        self.gnn = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, adj):
        """
        x:   [B, N, D]
        adj: [B, N, N]
        Output: [B, N, N]
        """
        batch_size, num_nodes, feat_dim = x.shape
        device = x.device
        output = torch.zeros(batch_size, num_nodes, num_nodes, device=device)

        for b in range(batch_size):
            xb = x[b]
            adjb = adj[b]
            hb = F.relu(self.node_proj(xb))

            edge_index_b = dense_to_sparse(adjb)[0]
            if edge_index_b.size(1) == 0:
                h_gnn = hb
            else:
                h_gnn = self.gnn(hb, edge_index_b)

            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and adjb[i, j]:
                        edge_feat = torch.cat([h_gnn[i], h_gnn[j]], dim=-1)
                        score = self.edge_mlp(edge_feat)
                        output[b, i, j] = score

        return output

def convert_batched_obs_to_list(obs, n_envs):
    observations = []
    for env in range(n_envs):
        obs_dict = {}
        for key in obs.keys():
            obs_dict[key] = obs[key][env]
        observations.append(obs_dict)
    return observations

def convert_batched_obs_to_tensor(obs, n_envs, n_timesteps):
    obs = convert_batched_obs_to_list(obs, n_envs)
    obs_tuples = [convert_obs_to_tensors(obs_dict, n_timesteps) for obs_dict in obs]
    agent_vecs, edge_index = zip(*obs_tuples)
    agent_vecs = torch.stack(agent_vecs)
    edge_index = torch.stack(edge_index)
    return agent_vecs, edge_index

def evaluate_parameters(model_arch: nn.Module, parameters: torch.Tensor, n_runs=8) -> List[float]:
    n_agents = 50
    n_timesteps = 16
    death_range = [10, 14]

    def generate_env(seed=None):
        return PairedKidneyDonationEnv(
            n_agents=n_agents,
            n_timesteps=n_timesteps,
            death_range=death_range,
            seed=seed
        )

    seeds = np.random.randint(0, 10000, n_runs)
    envs = [lambda: generate_env(seed=int(seed)) for seed in seeds]
    env = gym.vector.SyncVectorEnv(envs)

    # Prepare model clones for each parameter set
    models = []
    for param_vec in parameters:
        model = copy.deepcopy(model_arch)
        nn.utils.vector_to_parameters(param_vec, model.parameters())
        model.eval()
        models.append(model)

    rewards = []

    for model_index, model in tqdm(enumerate(models)):
        obs, _ = env.reset()
        done = False
        total_reward = np.zeros(n_runs)
        while not done:
            agent_vecs, edge_index = convert_batched_obs_to_tensor(obs, n_runs, n_timesteps)
            with torch.no_grad():
                action = model(agent_vecs, edge_index)
                action = action.cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = all(terminated) or all(truncated)
            total_reward += reward
        rewards.append(np.sum(total_reward) / n_runs)

    return rewards

model = SimpleBatchedPKEModel(hidden_dim=16)
num_parameters = sum(p.numel() for p in model.parameters())

num_epochs = 40
num_selected = 8
parameters = torch.randn(num_selected, num_parameters)
best_k = 2
num_children = (num_selected // best_k) - 1

thetas = np.linspace(0.1, 0.7, best_k)
thetas = torch.tensor(thetas)
thetas = thetas.unsqueeze(1)

alpha = 0.995

num_runs = 10

assert num_selected % best_k == 0, "num_selected must be divisible by best_k"

for epoch in tqdm(range(num_epochs)):
    rewards = evaluate_parameters(model, parameters, n_runs=num_runs)

    rewards_arr = np.array(rewards)
    iqr = np.percentile(rewards_arr, 75) - np.percentile(rewards_arr, 25)
    print(
        "Rewards statistics: ",
        "mean =", np.mean(rewards_arr),
        "std =", np.std(rewards_arr),
        "min =", np.min(rewards_arr),
        "max =", np.max(rewards_arr),
        "median =", np.median(rewards_arr),
        "25th percentile =", np.percentile(rewards_arr, 25),
        "75th percentile =", np.percentile(rewards_arr, 75),
        "iqr =", iqr
    )

    # select the best k parameters
    best_indices = np.argsort(rewards)[-best_k:]
    best_parameters = parameters[best_indices]
    best_parameters = torch.tensor(best_parameters)   
        
    new_parameters = torch.zeros_like(parameters)
    new_parameters[:best_k] = best_parameters
    for child in range(num_children):
        start = (child + 1) * best_k
        end = min(start + best_k, num_selected)
        deltas = torch.rand_like(best_parameters) * thetas 
        new_parameters[start:end] = best_parameters + deltas

    thetas = thetas * alpha
    parameters = new_parameters
print("Number of of parameters: ", num_parameters)

# Save the best model parameters
best_idx = np.argmax(rewards)
best_param_vec = parameters[best_idx]
best_model = copy.deepcopy(model)
nn.utils.vector_to_parameters(best_param_vec, best_model.parameters())
torch.save(best_model.state_dict(), "best_pke_model.pth")
print("Best model saved to best_pke_model.pth")