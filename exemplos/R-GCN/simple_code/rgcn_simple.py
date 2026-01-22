import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import RelGraphConv
from dgl.data import AIFBDataset


# ----------------------------------------------------
# 1. Load AIFB dataset (multi-relational knowledge graph)
# ----------------------------------------------------
dataset = AIFBDataset()
graph = dataset[0]

# node features
feat = graph.ndata['person']['feat']
num_classes = dataset.num_classes
num_rels = len(graph.canonical_etypes)


# ----------------------------------------------------
# 2. Define R-GCN Model (using built-in RelGraphConv)
# ----------------------------------------------------
class RGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels):
        super().__init__()
        self.layer1 = RelGraphConv(in_dim, h_dim, num_rels)
        self.layer2 = RelGraphConv(h_dim, out_dim, num_rels)

    def forward(self, g, features):
        h = self.layer1(g, features)
        h = F.relu(h)
        h = self.layer2(g, h)
        return h


# ----------------------------------------------------
# 3. Instantiate model
# ----------------------------------------------------
model = RGCN(
    in_dim=feat.shape[1],
    h_dim=16,
    out_dim=num_classes,
    num_rels=num_rels
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


# ----------------------------------------------------
# 4. Training function
# ----------------------------------------------------
def train():
    for epoch in range(50):
        model.train()
        logits = model(graph, feat)

        loss = F.cross_entropy(
            logits[dataset.train_mask],
            dataset.train_labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")


# ----------------------------------------------------
# 5. Evaluation
# ----------------------------------------------------
def evaluate():
    model.eval()
    logits = model(graph, feat)
    test_logits = logits[dataset.test_mask]
    test_labels = dataset.test_labels

    preds = torch.argmax(test_logits, dim=1)
    acc = (preds == test_labels).float().mean()

    print(f"\nTest Accuracy: {acc.item():.4f}")


# ----------------------------------------------------
# 6. Run everything
# ----------------------------------------------------
if __name__ == "__main__":
    train()
    evaluate()
