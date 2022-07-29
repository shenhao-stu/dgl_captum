import dgl
from torch import Tensor
import networkx as nx
from inspect import signature
import matplotlib.pyplot as plt
from math import sqrt

NID, EID = '_ID', '_ID'


def visualize_subgraph(graph: dgl.DGLGraph, node_idx: int, num_hops: int, edge_alpha: Tensor, node_alpha: Tensor = None, seed: int = 10, **kwargs):
    """
    Visualize a subgraph of the model.
    """

    assert edge_alpha.size(0) == graph.number_of_edges()

    # Only operate on a k-hop subgraph around `node_idx`.
    sg, _ = dgl.khop_in_subgraph(graph, node_idx, num_hops)
    subnode_idx = sg.ndata[NID].long()
    subedge_idx = sg.edata[EID].long()
    edge_alpha_subset = edge_alpha.gather(0, subedge_idx)
    # node_alpha_subset = node_alpha[subnode_idx]
    sg.edata['importance'] = edge_alpha_subset
    # sg.ndata['importance'] = node_alpha_subset

    # nx_g = sg.to_networkx(node_attrs=['importance'], edge_attrs=['importance'])
    nx_g = sg.to_networkx(edge_attrs=['importance'])
    mapping = {k: i for k, i in enumerate(subnode_idx.tolist())}
    nx_g = nx.relabel_nodes(nx_g, mapping)

    node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
    node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
    node_kwargs['node_size'] = kwargs.get('node_size') or 800
    node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

    label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
    label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
    label_kwargs['font_size'] = kwargs.get('font_size') or 10

    pos = nx.spring_layout(nx_g, seed=seed)
    ax = plt.gca()
    for source, target, data in nx_g.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                alpha=max(data['importance'].item(), 0.1),
                color='black',
                shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))

    node_color = [0] * sg.number_of_nodes()

    if node_alpha is None:
        nx.draw_networkx_nodes(nx_g, pos, node_color=node_color, **node_kwargs)
    else:
        assert node_alpha.size(0) == graph.number_of_nodes()
        node_alpha_subset = node_alpha[subnode_idx]
        assert ((node_alpha_subset >= 0) & (node_alpha_subset <= 1)).all()
        nx.draw_networkx_nodes(nx_g, pos, alpha=node_alpha_subset.tolist(
        ), node_color=node_color, **node_kwargs)

    nx.draw_networkx_labels(nx_g, pos, **label_kwargs)

    return ax, nx_g
