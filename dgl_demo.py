from functools import partial
from captum.attr import IntegratedGradients
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
from dgl.nn import GraphConv

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, in_feat, g, edge_weight=None):
        h = self.conv1(g, in_feat, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight=edge_weight)
        return h


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to(device)
g = g.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    # Forward
    logits = model(features, g)

    # Compute prediction
    pred = logits.argmax(1)

    # Compute loss
    # Note that you should only compute the losses of the nodes in the training set.
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])

    # Backward
    loss.backward()
    optimizer.step()

# 选取索引为10的节点进行可解释性分析
output_idx = 10
target = int(g.ndata['label'][output_idx])
print(target)


# Node explainability
ig = IntegratedGradients(partial(model.forward, g=g))
ig_attr_node = ig.attribute(g.ndata['feat'], target=target,
                            internal_batch_size=g.num_nodes(), n_steps=50)
print(ig_attr_node.shape)

# Scale attributions to [0, 1]:
ig_attr_node = ig_attr_node.abs().sum(dim=1)
ig_attr_node /= ig_attr_node.max()


def model_forward(edge_mask, g):
    out = model(g.ndata['feat'], g, edge_weight=edge_mask)
    return out


# Edge explainability
edge_mask = torch.ones(g.num_edges()).requires_grad_(True).to(device)
ig = IntegratedGradients(partial(model_forward, g=g))
ig_attr_edge = ig.attribute(edge_mask, target=target,
                            internal_batch_size=g.num_nodes(), n_steps=50)
print(ig_attr_edge.shape)
