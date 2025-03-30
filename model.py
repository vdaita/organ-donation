from torch_geometric.nn import GAT, global_mean_pool
from torch import nn
import torch
import torch.nn.functional as F

class PairedKidneyModel(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(PairedKidneyModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gat = GAT(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=num_layers)
        
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.item_priority_fc = nn.Linear(hidden_dim, 1)
        self.match_priority = nn.Linear(hidden_dim, 1)
        self.match_global = nn.Linear(hidden_dim, 1)

    def forward(self, adj_matrix, timestep):
        num_nodes = adj_matrix.size(0)
        
        x = torch.ones(num_nodes, self.hidden_dim, device=adj_matrix.device)
        edge_index = adj_matrix.nonzero().t().contiguous()
        node_embeddings = self.gat(x, edge_index)
        
        time_embed = self.time_embedding(timestep.view(1, 1)).squeeze(0)
        node_embeddings = node_embeddings + time_embed.unsqueeze(0)
        
        item_priority = torch.sigmoid(self.item_priority_fc(node_embeddings))
        graph_embedding = global_mean_pool(node_embeddings, batch=torch.zeros(num_nodes, dtype=torch.long, device=adj_matrix.device))

        match_priority = torch.sigmoid(self.match_priority(graph_embedding))
        match_global = torch.sigmoid(self.match_global(graph_embedding))
        
        return item_priority, match_priority, match_global