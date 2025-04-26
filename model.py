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

    B, N, _ = adj_matrix.shape
    timestep = timestep.unsqueeze(1).repeat(1, N)
    total_timesteps = total_timesteps.unsqueeze(1).repeat(1, N)

    time_since_arrival = (timestep - arrival) / (departure - arrival)
    time_since_start = timestep / total_timesteps

    multiples = []
    for k in [2, 4, 7]:
        multiples.append(time_since_start % k)
    multiples = torch.stack(multiples, dim=1)

    time_since_arrival = time_since_arrival.unsqueeze(1)
    time_since_start = time_since_start.unsqueeze(1)
    is_hard_to_match = is_hard_to_match.unsqueeze(1)

    print(
        "time_since_arrival: ", time_since_arrival.shape,
        "\ntime_since_start: ", time_since_start.shape,
        "\nmultiples: ", multiples.shape,
        "\nis_hard_to_match: ", is_hard_to_match.shape,
    )

    node_features = torch.cat([
        time_since_arrival,
        time_since_start,
        multiples,
        is_hard_to_match,
    ], dim=1)

    node_features = node_features.transpose(-1, -2)

    return node_features, adj_matrix, active_agents

class PairedKidneyBackbone(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(PairedKidneyBackbone, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
 
        self.embedding = nn.Sequential(
            nn.Linear(6, hidden_dim), # timestamp from distance of arrival to departure achieved, hard to match
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gat = GAT(in_channels=hidden_dim, hidden_channels=hidden_dim * 2, num_layers=num_layers, out_channels=hidden_dim)
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
        nn.init.xavier_uniform_(self.select_fc.weight)
        if self.select_fc.bias is not None:
            nn.init.zeros_(self.select_fc.bias)

    def forward(self, obs):
        device = self.get_device()
        node_features, adj_matrix, active_agents = get_feature_tensor(obs)

        B, N, _  = adj_matrix.shape

        print("node_features: ", node_features.shape)
        x = self.embedding(node_features)
        data_list = []

        for b in range(B):
            edges = adj_matrix[b].nonzero(as_tuple=False).t()
            data = Data(x=x[b], edge_index=edges)
            data_list.append(data)

        batch_data = Batch.from_data_list(data_list).to(device)
        x_gat = self.gat(batch_data.x, batch_data.edge_index)
        x_gat = x_gat.reshape(B, N, -1)
        x = x + x_gat
        for layer in self.ffprocess:
            x = x + F.relu(layer(x))
        x = x * active_agents.unsqueeze(-1)
        print("Backbone output: ", x.shape)
        return x

class PairedKidneyModel(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(PairedKidneyModel, self).__init__()
        
        self.backbone = PairedKidneyBackbone(hidden_dim, num_layers)
        self.select_fc = nn.Linear((hidden_dim), 1)
        
    def reset_parameters(self):
        self.backbone.reset_parameters()
        nn.init.xavier_uniform_(self.select_fc.weight)
        if self.select_fc.bias is not None:
            nn.init.zeros_(self.select_fc.bias)

    def forward(self, obs):
        x = self.backbone(obs)
        x = self.select_fc(x)
        x = F.sigmoid(x)
        print("After sigmoid and select_fc: ", x.shape)
        x = x.squeeze(-1)
        return x
    
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
        nn.init.xavier_uniform_(self.value_fc.weight)
        if self.value_fc.bias is not None:
            nn.init.zeros_(self.value_fc.bias)

    def forward(self, obs):
        x = self.backbone(obs)
        print("After backbone: ", x.shape)
        x = self.attention(x)
        print("After attention: ", x.shape)
        x = self.value_fc(x)
        print("After value_fc: ", x.shape)
        x = torch.relu(x)
        x = x.flatten()
        print("After flatten: ", x.shape)
        return x.flatten()