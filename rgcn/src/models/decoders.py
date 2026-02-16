import torch
import torch.nn as nn

class DistMultDecoder(nn.Module):
    def __init__(self, num_rels: int, h_dim: int):
        super().__init__()
        self.rel_emb = nn.Embedding(num_rels, h_dim)

        # Xavier init keeps relation vectors on a compatible scale.
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def score(self, h, r, t):
        # DistMult: <h * r, t>
        return torch.sum(h * r * t, dim=-1)

    def forward(self, node_emb, src, rel, dst):
        h = node_emb[src]
        r = self.rel_emb(rel)
        t = node_emb[dst]
        return self.score(h, r, t)