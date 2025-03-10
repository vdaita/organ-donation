from torch import nn
import torch

class DecisionTransformer(nn.Module):
    def __init__(self, n_attr=9, hidden_dim=128, n_heads=8, n_layers=2):
        super().__init__()
        
        self.patient_embedding = nn.Linear(n_attr, hidden_dim)
        self.token_type_embedding = nn.Embedding(2, hidden_dim)
        # don't really care for a positional embedding as order doesn't matter
        self.done_embedding = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=n_layers
        )
        self.weighter = nn.Linear(hidden_dim, 1)

        self._random_init()

    def _random_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, selected: torch.Tensor, current: torch.Tensor, patients: torch.Tensor):
        selected, current, patients = selected.to(torch.bool), current.to(torch.int32), patients.to(torch.float32)
        num_patients = patients.shape[-2]

        selected = torch.cat([selected, torch.Tensor([False]).to(selected.device)], dim=0).to(torch.bool)
        selected = selected.unsqueeze(0).repeat(num_patients + 1, 1)

        src_mask = torch.zeros((num_patients + 1, num_patients + 1)).to(selected.device)
        src_mask = src_mask.masked_fill(selected, float('-inf'))

        embedded_patients = self.patient_embedding(patients) + self.token_type_embedding(current)
        done_embedding_expanded = self.done_embedding.unsqueeze(0)
        embedded_patients = torch.cat([embedded_patients, done_embedding_expanded], dim=0)

        x = self.encoder(embedded_patients, mask=src_mask)

        x = self.weighter(x)
        x = x.squeeze(-1)

        current = torch.cat([current, torch.Tensor([0]).to(current.device)], dim=0).to(torch.int32) # extend this to the last element
        current_mask = current * -1e9
        x = x + current_mask

        x = torch.softmax(x, dim=-1)
        return x