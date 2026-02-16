import torch
import dgl
from dgl.nn.pytorch import RelGraphConv

def test_second_layer_behavior():
    torch.manual_seed(42)
    num_nodes = 10
    h_dim = 8
    num_rels = 3
    num_bases = 3
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
    # Instantiate the same layer configuration as the model
    layer = RelGraphConv(
        in_feat=h_dim,
        out_feat=h_dim,
        num_rels=num_rels,
        regularizer="basis",
        num_bases=num_bases,
        self_loop=False,
        dropout=0.0
    )
    # Forward
    out = layer(g, x, g.edata['etype'], g.edata['norm'])
    print("Input for the second layer (x):\n", x)
    print("Second layer output:\n", out)
    print("Output min:", out.min().item(), "max:", out.max().item(), "mean:", out.mean().item())

if __name__ == "__main__":
    test_second_layer_behavior()