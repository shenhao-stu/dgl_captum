import os.path as osp

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from captum.attr import IntegratedGradients
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Explainer, GCNConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Planetoid')
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


def model_forward(edge_mask, data):
    out = model(data.x, data.edge_index, edge_mask)
    return out


def explain(data, target=0):
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    ig = IntegratedGradients(model_forward)
    mask = ig.attribute(input_mask, target=target,
                        additional_forward_args=(data,),
                        internal_batch_size=data.edge_index.shape[1])
    # Scale attributions to [0, 1]:
    edge_mask = mask.abs()
    edge_mask /= edge_mask.max()
    return edge_mask


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    log_logits = model(data.x, data.edge_index)
    loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

output_idx = 10
target = int(data.y[output_idx])

# Edge explainability
# ===================

ig_attr_edge = explain(data, target=0)

# Visualize absolute values of attributions:
explainer = Explainer(model)
ax, G = explainer.visualize_subgraph(output_idx, data.edge_index, ig_attr_edge)
plt.show()
