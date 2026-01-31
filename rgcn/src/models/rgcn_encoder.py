import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

class RGCNEncoder(nn.Module):
    def __init__(self, num_nodes: int, h_dim: int, num_rels: int, num_bases: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.num_rels = num_rels

        self.emb = nn.Embedding(num_nodes, h_dim)
        # INICIALIZAÇÃO: Garante que os embeddings não ficam a zeros
        with torch.no_grad():
            nn.init.xavier_uniform_(self.emb.weight)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                RelGraphConv(
                    in_feat=h_dim,
                    out_feat=h_dim,
                    num_rels=num_rels,
                    regularizer="basis",
                    num_bases=num_bases,
                    self_loop=True,  # OBRIGATÓRIO: Permite que o fármaco mantenha o seu próprio embedding
                    dropout=dropout
                )
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, etype):
        x = self.emb.weight
        for i, layer in enumerate(self.layers):
            x = layer(g, x, etype)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x
