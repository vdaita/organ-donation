from torch_geometric.nn import GAT, global_mean_pool
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class PairedKidneyModel(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(PairedKidneyModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Sequential(
            nn.Linear(2, hidden_dim), # timestamp from distance of arrival to departure achieved, hard to match
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gat = GAT(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=num_layers)
        self.select_fc = nn.Linear((hidden_dim), 1)
    
        self.device = next(self.parameters()).device
        

    def reset_parameters(self):
        for layer in self.embedding:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.gat.reset_parameters()
        nn.init.xavier_uniform_(self.select_fc.weight)
        if self.select_fc.bias is not None:
            nn.init.zeros_(self.select_fc.bias)

    def forward(self, obs):
        adj_matrix = torch.Tensor(obs["adjacency_matrix"]).to(self.device)
        timestep = obs["timestep"]
        arrival = torch.Tensor(obs["arrivals"]).to(self.device)
        departure = torch.Tensor(obs["departures"]).to(self.device)
        is_hard_to_match = torch.Tensor(obs["is_hard_to_match"]).to(self.device)
        active_agents = torch.Tensor(obs["active_agents"]).to(self.device)

        if torch.sum(active_agents) == 0:
            return torch.zeros((adj_matrix.size(0), 1), device=adj_matrix.device)
        
        masked_indices = torch.nonzero(active_agents, as_tuple=False).squeeze()
        num_active = masked_indices.size(0)            
        relevant_arrivals = arrival[masked_indices]
        relevant_departures = departure[masked_indices]
        timestep_expanded = torch.full((num_active, ), timestep, device=adj_matrix.device)
        relevant_progress = (timestep_expanded - relevant_arrivals) / (relevant_departures - relevant_arrivals)
        in_data = torch.concat([
            relevant_progress.unsqueeze(1), 
            is_hard_to_match[masked_indices].unsqueeze(1)
        ], dim=1)

        x = self.embedding(in_data)
        
        adj_matrix_revised = adj_matrix[np.ix_(masked_indices, masked_indices)]
        edge_index = adj_matrix_revised.nonzero(as_tuple=False).t()

        x = x + self.gat(x, edge_index) # adding residual
        x = F.layer_norm(x, x.size()[1:]) # layer normalization

        x = self.select_fc(x)
        x = F.sigmoid(x)

        ret_nodes = torch.zeros((adj_matrix.size(0), 1), device=adj_matrix.device)
        ret_nodes[masked_indices] = x
        return ret_nodes

class PairedKidneyCriticModel(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(PairedKidneyCriticModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Sequential(
            nn.Linear(2, hidden_dim), # timestamp from distance of arrival to departure achieved, hard to match
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gat = GAT(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=num_layers)
        self.value_fc = nn.Linear(hidden_dim, 1)
    
        self.device = next(self.parameters()).device
    
    def reset_parameters(self):
        for layer in self.embedding:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.gat.reset_parameters()
        nn.init.xavier_uniform_(self.value_fc.weight)
        if self.value_fc.bias is not None:
            nn.init.zeros_(self.value_fc.bias)

    def forward(self, obs):
        adj_matrix = torch.Tensor(obs["adjacency_matrix"]).to(self.device)
        timestep = obs["timestep"]
        arrival = torch.Tensor(obs["arrivals"]).to(self.device)
        departure = torch.Tensor(obs["departures"]).to(self.device)
        is_hard_to_match = torch.Tensor(obs["is_hard_to_match"]).to(self.device)
        active_agents = torch.Tensor(obs["active_agents"]).to(self.device)

        if torch.sum(active_agents) == 0:
            return 0
        
        masked_indices = torch.nonzero(active_agents, as_tuple=False).squeeze()
        num_active = masked_indices.size(0)            
        relevant_arrivals = arrival[masked_indices]
        relevant_departures = departure[masked_indices]
        timestep_expanded = torch.full((num_active, ), timestep, device=adj_matrix.device)
        relevant_progress = (timestep_expanded - relevant_arrivals) / (relevant_departures - relevant_arrivals)
        in_data = torch.concat([
            relevant_progress.unsqueeze(1), 
            is_hard_to_match[masked_indices].unsqueeze(1)
        ], dim=1)

        x = self.embedding(in_data)
        
        adj_matrix_revised = adj_matrix[np.ix_(masked_indices, masked_indices)]
        edge_index = adj_matrix_revised.nonzero(as_tuple=False).t()

        x = x + self.gat(x, edge_index) # adding residual
        x = F.layer_norm(x, x.size()[1:]) # layer normalization
        
        batch = torch.zeros_like(relevant_arrivals, dtype=torch.long, device=adj_matrix.device)
        x = global_mean_pool(x, batch=batch)
        
        x = self.value_fc(x)
        x = F.relu(x)
        return x.squeeze()