import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleGraphEncoder(nn.Module):
    """Simple graph encoder that processes nodes and their connections"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers (simplified graph convolution)
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, obs):
        # Extract features from observation
        adj = torch.FloatTensor(obs["adjacency_matrix"])
        is_hard = torch.FloatTensor(obs["is_hard_to_match"]).unsqueeze(-1)
        active = torch.FloatTensor(obs["active_agents"]).unsqueeze(-1)
        time_features = torch.FloatTensor([
            obs["timestep"] / obs["total_timesteps"],  # normalized timestep
            obs["timestep"] % 2 / 2,  # cyclical feature
            obs["timestep"] % 4 / 4   # cyclical feature
        ]).unsqueeze(0).repeat(len(active), 1)
        
        # Create node features
        node_features = torch.cat([is_hard, active, time_features], dim=-1)
        
        # Encode node features
        x = self.node_encoder(node_features)
        
        # Simple message passing (two rounds)
        x1 = F.relu(self.conv1(x))
        x1 = x1 + torch.matmul(adj, x1) / (torch.sum(adj, dim=1, keepdim=True) + 1e-6)
        
        x2 = F.relu(self.conv2(x1))
        x2 = x2 + torch.matmul(adj, x2) / (torch.sum(adj, dim=1, keepdim=True) + 1e-6)
        
        # Mask inactive nodes
        node_embeddings = x2 * active
        
        return node_embeddings, adj, active

class SimpleEdgePredictor(nn.Module):
    """Simple model that predicts edge selection probabilities"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.encoder = SimpleGraphEncoder(hidden_dim)
        self.edge_predictor = nn.Bilinear(hidden_dim, hidden_dim, 1)
        
    def forward(self, obs):
        # Get node embeddings
        node_emb, adj_matrix, active = self.encoder(obs)
        
        # Predict edge probabilities using bilinear layer
        batch_size, n_nodes = 1, node_emb.shape[0]
        if len(node_emb.shape) > 2:
            batch_size = node_emb.shape[0]
        
        edge_scores = []
        for b in range(batch_size):
            n_emb = node_emb[b] if batch_size > 1 else node_emb
            # Compute pairwise scores using bilinear layer
            i_indices, j_indices = torch.where(torch.ones(n_nodes, n_nodes))
            scores = self.edge_predictor(
                n_emb[i_indices], 
                n_emb[j_indices]
            ).reshape(n_nodes, n_nodes)
            
            # Apply masks
            valid_edges = adj_matrix[b] if batch_size > 1 else adj_matrix
            a_mask = torch.outer(active[b], active[b]) if batch_size > 1 else torch.outer(active, active)
            
            # Get final probabilities
            probs = torch.sigmoid(scores) * valid_edges * a_mask
            edge_scores.append(probs)
            
        return torch.stack(edge_scores, dim=0) if batch_size > 1 else edge_scores[0]
        
class SimpleCritic(nn.Module):
    """Simple critic network for PPO"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.encoder = SimpleGraphEncoder(hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs):
        node_emb, _, active = self.encoder(obs)
        # Global pooling
        global_emb = torch.mean(node_emb, dim=0) if len(node_emb.shape) <= 2 else torch.mean(node_emb, dim=1)
        # Value prediction
        value = self.value_head(global_emb).squeeze(-1)
        return value
