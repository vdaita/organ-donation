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
        self.select_fc = nn.Linear((hidden_dim + 1), 1)
        

    def reset_parameters(self):
        for layer in self.embedding:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.gat.reset_parameters()
        nn.init.xavier_uniform_(self.select_fc.weight)
        if self.select_fc.bias is not None:
            nn.init.zeros_(self.select_fc.bias)

    def forward(self, adj_matrix, timestep, arrival, departure, is_hard_to_match, total_timesteps, mask):    
        try:
            masked_indices = torch.nonzero(mask, as_tuple=False).squeeze()
            num_active = masked_indices.size(0)            
            relevant_nodes = adj_matrix[masked_indices]
            relevant_arrivals = arrival[masked_indices]
            relevant_departures = departure[masked_indices]
            # print("timestamp: ", timestep)
            timestep_expanded = torch.full((num_active, ), timestep, device=adj_matrix.device)

            # print("Input values to next stage: ", " nodes: ", relevant_nodes,  " arrivals: ", relevant_arrivals, " departures: ", relevant_departures, " timestep: ", timestep_expanded)    

            relevant_progress = (timestep_expanded - relevant_arrivals) / (relevant_departures - relevant_arrivals)
            
            # print("Relevant progress: ", relevant_progress.shape, relevant_progress)
            # print("Is hard to match: ", is_hard_to_match[masked_indices].shape, is_hard_to_match[masked_indices])
            in_data = torch.concat([
                relevant_progress.unsqueeze(1), 
                is_hard_to_match[masked_indices].unsqueeze(1)
            ], dim=1)
            # print("In data: ", in_data, in_data.shape)

            x = self.embedding(in_data)
            
            adj_matrix_revised = adj_matrix[np.ix_(masked_indices, masked_indices)]
            # print("X: ", x.shape)
            # print("Adjacency matrix revised: ", adj_matrix_revised, adj_matrix_revised.shape)
            edge_index = adj_matrix_revised.nonzero(as_tuple=False).t()

            x = x + self.gat(x, edge_index) # adding residual
            x = F.layer_norm(x, x.size()[1:]) # layer normalization

            time_context = torch.full((num_active, 1), (timestep / total_timesteps), device=adj_matrix.device)
            x = torch.cat([x, time_context], dim=1)

            x = self.select_fc(x)


            x = F.sigmoid(x)

            # print("X output: ", x)

            ret_nodes = torch.zeros((adj_matrix.size(0), 1), device=adj_matrix.device)
            ret_nodes[masked_indices] = x
            # print("Returned nodes: ", ret_nodes)
        except Exception as e:
            print("Error: ", e)
            ret_nodes = torch.zeros((adj_matrix.size(0), 1), device=adj_matrix.device)
        return ret_nodes
