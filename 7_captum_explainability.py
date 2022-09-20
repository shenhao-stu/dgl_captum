"""
Explaining Graph Neural Network with Captum
===========================================

By the end of this tutorial, you will be able to:

- Understand the basic concepts of GNN explainability
- Use Captum to explain the node classification task
- Find the most important nodes and edges to the model
- Visualize the node-centered subgraphs

(Time estimate: 20 minutes)

"""

######################################################################
# Overviews of GNN Explainability
# -------------------------------
# In recent years, research on the interpretability of deep learning models has
# made significant progress. Compared to cv and nlp domains, there is less
# research and application of graph model interpretability, yet it is the key
# to understanding deep graph neural networks. Generally, GNN explanation
# research often starts with the following tasks: ** Which input edges are more
# important? Which input nodes are more important? **

# Node-centered subgraphs play a critical role in analyzing GNNs. The k-hop
# subgraph of a node fully determines the information a k-layer GNN exploits
# to generate its final node representation. Many GNN explanation methods
# provide explanations by extracting a subgraph and assigning importance
# weights to the nodes and edges of it. We will visualize node-centered
# weighted subgraphs through DGL's built-in functions. This is beneficial
# for debugging and understanding GNNs and GNN explanation methods.

# For this demonstration, we will use IntegratedGradients from `Captum <https://github.com/pytorch/captum>`__ to
# explain the predictions of a graph convolutional network (GCN).
# Specifically, we try to find the most important nodes and edges to the model in node classification.
# Captum is a model interpretability and understanding library for PyTorch. You can install it with

# .. code:: bash
#
# pip install captum
#


######################################################################
# Loading Cora Dataset
# --------------------
#
# First, we load DGL’s built-in Cora dataset and retrieve its graph structure, node labels (classes) and the number of node classes.
# 

# Install and import required packages.
import dgl
import dgl.data
# The Cora dataset used in this tutorial only consists of one single graph.
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

######################################################################
# Define the model
# ----------------
# Then, we will build a two-layer Graph Convolutional Network (GCN).
# Each layer computes new node representations by aggregating neighbor information.
# What's more, we use GraphConv which supports ``edge_weight`` as a parameter to calculate
# the importance of the edge in the edge explainability task.
# 

from dgl.nn import GraphConv
# Define a class for GCN
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

######################################################################
# Training the model
# ------------------
#

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to(device)
g = g.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']

for epoch in range(1,201):
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

######################################################################
# Explaining the predictions
# --------------------------
# First, we will complete the task of which input nodes are more important.
# We attribute the model predictions to the input node features with IntegratedGradients.
#

# Select the node with index 10 for interpretability analysis
output_idx = 10
target = int(g.ndata['label'][output_idx])
print(target)

# Since the ``IntergratedGradients`` method only allows one argument to be passed, we use ``partial`` function to pass the default value to the forward function.
#

# import captum
from captum.attr import IntegratedGradients
from functools import partial

# Node explainability
ig = IntegratedGradients(partial(model.forward, g=g))
# Attribute the predictions for node class 0 to the input features
ig_attr_node = ig.attribute(g.ndata['feat'], target=target,
                            internal_batch_size=g.num_nodes(), n_steps=50)
print(ig_attr_node.shape)


# We compute the node importance weights from the input feature weights and normalize them.
#

# Scale attributions to [0, 1]:
ig_attr_node = ig_attr_node.abs().sum(dim=1)
ig_attr_node /= ig_attr_node.max()


# We visualize node-centered weighted subgraphs through DGL's built-in functions.
#

# Visualize
from utility import visualize_subgraph
import matplotlib.pyplot as plt

num_hops = 2
ax, nx_g = visualize_subgraph(g, output_idx, num_hops, node_alpha=ig_attr_node)
plt.show()

# <img src="README.assets/Figure_1.png" style="zoom:50%;" />
# Then, we will complete the task of which input edges are more important.
# To apply the IntergratedGradients method, we redefine the forward function
#

def model_forward(edge_mask, g):
    out = model(g.ndata['feat'],g,edge_weight=edge_mask)
    return out

# Edge explainability
edge_mask = torch.ones(g.num_edges()).requires_grad_(True).to(device)
ig = IntegratedGradients(partial(model_forward, g=g))
ig_attr_edge = ig.attribute(edge_mask, target=target,
                            internal_batch_size=g.num_nodes(), n_steps=50)
print(ig_attr_edge.shape)

# We compute the node importance weights from the input feature weights and normalize them.
#

# Scale attributions to [0, 1]:
ig_attr_edge = ig_attr_edge.abs()
ig_attr_edge /= ig_attr_edge.max()

# We visualize node-centered weighted subgraphs through DGL's built-in functions.
#

# Visualize
ax, nx_g = visualize_subgraph(g, output_idx, num_hops, edge_alpha=ig_attr_edge)
plt.show()

# <img src="README.assets/Figure_2.png" style="zoom:50%;" />
#

# Visualize node and edge explainability
ax, nx_g = visualize_subgraph(g, output_idx, num_hops, node_alpha=ig_attr_node, edge_alpha=ig_attr_edge)
plt.show()

# <img src="README.assets/Figure_3.png" style="zoom:50%;" />
#

