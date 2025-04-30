from torch_geometric.nn import GAT, GATv2Conv, global_mean_pool, GlobalAttention
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_mean

def get_feature_tensor(obs):
    # there are two ways to represent this: a list of dictionaries that have the same keys, or a dictionary of lists. code simplifies when both are comaptible
    if isinstance(obs, list):
        obs = {k: [d[k] for d in obs] for k in obs[0].keys()}

    adj_matrix = torch.tensor(obs["adjacency_matrix"])        
    arrival = torch.tensor(obs["arrivals"])
    departure = torch.tensor(obs["departures"])
    is_hard_to_match = torch.tensor(obs["is_hard_to_match"])
    active_agents = torch.tensor(obs["active_agents"])
    timestep = torch.tensor(obs["timestep"])
    total_timesteps = torch.tensor(obs["total_timesteps"])

    if len(adj_matrix.shape) == 2:
        adj_matrix = adj_matrix.unsqueeze(0)
        arrival = arrival.unsqueeze(0)
        departure = departure.unsqueeze(0)
        is_hard_to_match = is_hard_to_match.unsqueeze(0)
        active_agents = active_agents.unsqueeze(0)
        timestep = timestep.unsqueeze(0)
        total_timesteps = total_timesteps.unsqueeze(0)

    B, N, _ = adj_matrix.shape
    timestep = timestep.unsqueeze(1).repeat(1, N)
    total_timesteps = total_timesteps.unsqueeze(1).repeat(1, N)

    active_agents_count = torch.tensor(torch.sum(active_agents, dim=1))
    active_agents_count = active_agents_count / active_agents.shape[1]
    active_agents_count = active_agents_count.unsqueeze(1)
    active_agents_count = active_agents_count.unsqueeze(1)
    active_agents_count = active_agents_count.repeat(1, 1, N)

    time_since_arrival = (timestep - arrival) / (departure - arrival)
    time_since_start = timestep / total_timesteps

    multiples = []
    for k in [2, 4, 7]:
        multiples.append(time_since_start % k)
    multiples = torch.stack(multiples, dim=1)

    time_since_arrival = time_since_arrival.unsqueeze(1)
    time_since_start = time_since_start.unsqueeze(1)
    is_hard_to_match = is_hard_to_match.unsqueeze(1)

    node_features = torch.cat([
        time_since_arrival,
        time_since_start,
        multiples,
        is_hard_to_match,
        active_agents_count
    ], dim=1)

    node_features = node_features.transpose(-1, -2)

    return node_features, adj_matrix, active_agents

class PairedKidneyBackbone(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(PairedKidneyBackbone, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
 
        self.embedding = nn.Sequential(
            nn.Linear(7, hidden_dim), # timestamp from distance of arrival to departure achieved, hard to match
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gat = GAT(in_channels=hidden_dim, hidden_channels=hidden_dim * 2, num_layers=num_layers, out_channels=hidden_dim, norm="LayerNorm")
        self.ffprocess = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ])
    
    def get_device(self):
        return next(self.parameters()).device
        
    def reset_parameters(self):
        for layer in self.embedding:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        self.gat.reset_parameters()
        
        for layer in self.ffprocess:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, obs):
        device = self.get_device()
        node_features, adj_matrix, active_agents = get_feature_tensor(obs)

        B, N, _  = adj_matrix.shape

        x = self.embedding(node_features)
        data_list = []

        for b in range(B):
            edges = adj_matrix[b].nonzero(as_tuple=False).t()
            data = Data(x=x[b], edge_index=edges)
            data_list.append(data)

        batch_data = Batch.from_data_list(data_list).to(device)
        x_gat = self.gat(batch_data.x, batch_data.edge_index)
        x_gat = x_gat.reshape(B, N, -1)
        x = x + 0.25 * x_gat
        for layer in self.ffprocess:
            x = x + F.relu(layer(x))
        x = x * active_agents.unsqueeze(-1)
        return x, active_agents

class PairedKidneyModel(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(PairedKidneyModel, self).__init__()
        
        self.backbone = PairedKidneyBackbone(hidden_dim, num_layers)
        self.edge_proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.edge_proj2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.edge_score = nn.Linear(hidden_dim, 1)
        
    def reset_parameters(self):
        self.backbone.reset_parameters()
        nn.init.xavier_uniform_(self.edge_proj1.weight)
        nn.init.xavier_uniform_(self.edge_proj2.weight)
        nn.init.xavier_uniform_(self.edge_score.weight)
        if self.edge_score.bias is not None:
            self.edge_score.bias.data.fill_(0.0)

    def forward(self, obs):
        x, active_agents = self.backbone(obs)
        
        # Handle adjacency matrix extraction properly
        if isinstance(obs, dict):
            adj_matrix = torch.tensor(obs["adjacency_matrix"]).to(x.device)
        else:
            # Handle case where obs is a list of dictionaries
            adj_matrix = torch.stack([torch.tensor(o["adjacency_matrix"]) for o in obs]).to(x.device)
            
        if len(adj_matrix.shape) == 2:
            adj_matrix = adj_matrix.unsqueeze(0)
        
        batch_size, n_nodes = x.shape[0], x.shape[1]
        
        x_i = self.edge_proj1(x)
        x_j = self.edge_proj2(x)
        
        edge_probs = []
        for b in range(batch_size):
            x_i_expanded = x_i[b].unsqueeze(1).expand(-1, n_nodes, -1)
            x_j_expanded = x_j[b].unsqueeze(0).expand(n_nodes, -1, -1)
            
            edge_features = torch.cat([x_i_expanded, x_j_expanded], dim=-1)
            
            scores = self.edge_score(edge_features).squeeze(-1)
            
            probs = torch.sigmoid(scores) * adj_matrix[b]
            
            active_mask = torch.outer(active_agents[b], active_agents[b])
            probs = probs * active_mask
            
            edge_probs.append(probs)
        
        edge_probs = torch.stack(edge_probs, dim=0)
        
        if edge_probs.shape[0] == 1:
            edge_probs = edge_probs.squeeze(0)
            
        return edge_probs
    
class PairedKidneyCriticModel(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(PairedKidneyCriticModel, self).__init__()

        self.backbone = PairedKidneyBackbone(hidden_dim, num_layers)
        self.attention = GlobalAttention(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ))
        self.value_fc = nn.Linear(hidden_dim, 1)
    
    def reset_parameters(self):
        self.backbone.reset_parameters()
        
        for module in self.attention.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        
        nn.init.xavier_uniform_(self.value_fc.weight)
        if self.value_fc.bias is not None:
            nn.init.zeros_(self.value_fc.bias)

    def forward(self, obs):
        x, active_agents = self.backbone(obs)
        x = self.attention(x)
        x = self.value_fc(x)
        x = torch.relu(x)
        x = x.flatten()
        return x