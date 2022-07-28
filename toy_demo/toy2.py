import dgl
import dgl.nn as dglnn
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from functools import partial

w = dglnn.EdgeWeightNorm()
m = dglnn.GraphConv(20, 30, norm='none')

gs = []
for _ in range(5):
    g = dgl.graph((torch.randint(0, 20, (100,)),
                  torch.randint(0, 20, (100,))), num_nodes=20)
    g = dgl.add_reverse_edges(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    gs.append(g)
g = dgl.batch(gs)

edge_weight = (torch.randn(g.num_edges()) **
               2).requires_grad_()    # weighted graph
# node feature inputs
x = torch.randn(g.num_nodes(), 20).requires_grad_()


def forward(x, edge_weight, g):
    norm = w(g, edge_weight)
    return F.relu(m(g, x, edge_weight=norm))


ig = IntegratedGradients(partial(forward, edge_weight=edge_weight, g=g))
ig.attribute(x, target=0, internal_batch_size=g.num_nodes(), n_steps=50)
