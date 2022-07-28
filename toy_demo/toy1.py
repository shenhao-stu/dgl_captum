import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from dgl.nn.pytorch import GraphConv
from functools import partial

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, out_feats)
        self.conv2 = GraphConv(out_feats, out_feats)

    def forward(self, nfeats, g, nid=None):
        h = self.conv1(g, nfeats)
        h = F.relu(h)
        if nid is not None:
            return self.conv2(g, h)[nid:nid+1]
        else:
            return self.conv2(g, h)


g = dgl.graph(([1, 2, 3, 4, 4], [0, 1, 2, 3, 4]))
feat_size = 5
nfeats = torch.randn(g.num_nodes(), feat_size)
num_cls = 10
model = GCN(feat_size, num_cls)
ig = IntegratedGradients(partial(model.forward, g=g))
result = ig.attribute(nfeats, target=0, internal_batch_size=g.num_nodes(), n_steps=50)
res = result.detach().numpy()

print(np.mean(res,axis=0))