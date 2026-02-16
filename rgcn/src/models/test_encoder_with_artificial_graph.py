import torch
import dgl
from rgcn.src.models.rgcn_encoder import RGCNEncoder

def test_encoder_with_artificial_graph():
    torch.manual_seed(42)
    num_nodes = 10
    h_dim = 8
    num_rels = 3
    num_bases = 3
    num_layers = 2
    # Simple graph
    src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    etype = torch.randint(0, num_rels, (10,))
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.edata['etype'] = etype
    g.edata['norm'] = torch.ones(10, 1)
    # Input for the second layer: many zeros, a few positives
    x = torch.zeros(num_nodes, h_dim)
    x[0, 0] = 1.0
    x[1, 1] = 2.0
    x[2, 2] = 3.0
    x[3, 3] = 4.0
    x[4, 4] = 5.0
    # Instantiate encoder with the same settings as the real model
    encoder = RGCNEncoder(num_nodes, h_dim, num_rels, num_bases, num_layers, dropout=0.0)
    # Replace embeddings with the artificial input
    with torch.no_grad():
        encoder.emb.weight.copy_(x)
    # Forward
    out = encoder(g, g.edata['etype'])
    print("Artificial input (x):\n", x)
    print("Encoder output:\n", out)
    print("Output min:", out.min().item(), "max:", out.max().item(), "mean:", out.mean().item())

if __name__ == "__main__":
    test_encoder_with_artificial_graph()