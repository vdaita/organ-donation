from torch_geometric.nn import GAT, GATv2Conv, global_mean_pool
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class PairedKidneyBackbone(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(PairedKidneyBackbone, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Sequential(
            nn.Linear(3, hidden_dim), # timestamp from distance of arrival to departure achieved, hard to match
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gat = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False)
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

        adj_matrix = torch.Tensor(obs["adjacency_matrix"]).to(device)
        timestep = obs["timestep"]
        arrival = torch.Tensor(obs["arrivals"]).to(device)
        departure = torch.Tensor(obs["departures"]).to(device)
        is_hard_to_match = torch.Tensor(obs["is_hard_to_match"]).to(device)
        active_agents = torch.Tensor(obs["active_agents"]).to(device)

        if torch.sum(active_agents) == 0:
            return torch.zeros((adj_matrix.size(0), 1), device=adj_matrix.device)
        
        masked_indices = torch.nonzero(active_agents, as_tuple=False).squeeze(1)
        
        if masked_indices.dim() == 0:
            masked_indices = masked_indices.unsqueeze(0)
            
        num_active = masked_indices.size(0)            
        relevant_arrivals = arrival[masked_indices]
        relevant_departures = departure[masked_indices]
        timestep_expanded = torch.full((num_active, ), timestep, device=adj_matrix.device)
        
        # Add epsilon to avoid division by zero
        time_diff = relevant_departures - relevant_arrivals
        time_diff = torch.clamp(time_diff, min=1.0)  # Avoid division by zero
        relevant_progress = (timestep_expanded - relevant_arrivals) / time_diff

        time_until_departure = relevant_departures - timestep_expanded
        about_to_leave = (time_until_departure <= 1).float()
        
        in_data = torch.concat([
            relevant_progress.unsqueeze(1), 
            is_hard_to_match[masked_indices].unsqueeze(1),
            about_to_leave.unsqueeze(1)
        ], dim=1)

        x = self.embedding(in_data)
        
        adj_matrix_revised = adj_matrix[np.ix_(masked_indices, masked_indices)]
        edge_index = adj_matrix_revised.nonzero(as_tuple=False).t()
        
        # Only perform GAT if there are edges in the graph
        if edge_index.shape[1] > 0:
            x = x + self.gat(x, edge_index) # adding residual

        x = F.layer_norm(x, x.size()[1:]) 
        for layer in self.ffprocess:
            x = x + F.relu(layer(x))

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
        adj_matrix = torch.Tensor(obs["adjacency_matrix"]).to(self.backbone.get_device())
        active_agents = torch.Tensor(obs["active_agents"]).to(self.backbone.get_device())
        if torch.sum(active_agents) == 0:
            return torch.zeros((adj_matrix.size(0), 1), device=adj_matrix.device)
        masked_indices = torch.nonzero(active_agents, as_tuple=False).squeeze(1)
        
        x = self.backbone(obs)
        x = self.select_fc(x)
        x = F.sigmoid(x)

        ret_nodes = torch.zeros((adj_matrix.size(0), 1), device=adj_matrix.device)
        ret_nodes[masked_indices] = x
        return ret_nodes

class PairedKidneyCriticModel(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(PairedKidneyCriticModel, self).__init__()

        self.backbone = PairedKidneyBackbone(hidden_dim, num_layers)
        self.value_fc = nn.Linear(hidden_dim, 1)
    
    def reset_parameters(self):
        self.backbone.reset_parameters()
        nn.init.xavier_uniform_(self.value_fc.weight)
        if self.value_fc.bias is not None:
            nn.init.zeros_(self.value_fc.bias)

    def forward(self, obs):
        adj_matrix = torch.Tensor(obs["adjacency_matrix"]).to(self.backbone.get_device())
        relevant_arrivals = torch.Tensor(obs["arrivals"]).to(self.backbone.get_device())
        active_agents = torch.Tensor(obs["active_agents"]).to(self.backbone.get_device())

        if torch.sum(active_agents) == 0:
            return 0
        
        masked_indices = torch.nonzero(active_agents, as_tuple=False).squeeze(1)
        
        if masked_indices.dim() == 0:
            masked_indices = masked_indices.unsqueeze(0)
        
        relevant_arrivals = relevant_arrivals[masked_indices]
            
        x = self.backbone(obs)
        batch = torch.zeros_like(relevant_arrivals, dtype=torch.long, device=adj_matrix.device)
        x = global_mean_pool(x, batch=batch)
        
        x = self.value_fc(x)
        return x.squeeze()