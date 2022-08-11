import dgl.data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from dgl.nn.pytorch import GraphConv

from utility import visualize_subgraph

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
g = dgl.add_self_loop(g)


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, in_feat, g, edge_weight=None, nid=None):
        h = self.conv1(g, in_feat, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight=edge_weight)
        # nid is used to identify the target node
        if nid is None:
            return h
        else:
            return h[nid:nid + 1]


features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(features.shape[1], 16, dataset.num_classes).to(device)
g = g.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    # Forward
    logits = model(features, g)

    # Compute loss
    # Note that you should only compute the losses of the nodes in the training set.
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])

    # Backward
    loss.backward()
    optimizer.step()

# Select the node with index 10 for interpretability analysis
output_idx = 10
target = int(labels[output_idx])
print(target)


def combine_model_forward(in_feat, edge_mask, g, nid):
    out = model(in_feat, g, edge_weight=edge_mask, nid=nid)
    return out


edge_mask = torch.ones(g.num_edges()).requires_grad_(True).to(device)
ig = IntegratedGradients(combine_model_forward)
ig_attr_node, ig_attr_edge = ig.attribute((features, edge_mask), target=target, additional_forward_args=(g, output_idx),
                                          internal_batch_size=1, n_steps=50)

# Scale attributions to [0, 1]:
ig_attr_node = ig_attr_node.abs().sum(dim=1)
ig_attr_node /= ig_attr_node.max()
ig_attr_edge = ig_attr_edge.abs()
ig_attr_edge /= ig_attr_edge.max()

# Visualize node and edge explainability
num_hops = 2
ax, nx_g = visualize_subgraph(
    g, output_idx, num_hops, node_alpha=ig_attr_node, edge_alpha=ig_attr_edge)
plt.show()
