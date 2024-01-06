import numpy as np

import src.graphs_networx as gnx
from src.generate_data import RawData, RawDataConfig
from src.visualizers import visualise_2d_networkx


def internal_interconnectivity(graph, cid):
    c_edges = gnx.cluster_bisection_edges(graph, cid)
    sum_weights = gnx.sum_weight_edges(c_edges, graph)
    return sum_weights

def internal_closeness(graph, cid):
    subgraph = gnx.get_cluster_subgraph(graph, cid)
    sg_edges = subgraph.edges()
    sum_weights = gnx.sum_weight_edges(sg_edges, subgraph)
    return sum_weights

def relative_interconnectivity(graph, cid1, cid2):
    partitions = (gnx.get_cluster_nodes(graph, cid1), gnx.get_cluster_nodes(graph, cid2))
    edges = gnx.connection_edges_between(partitions, graph)
    ri_all = gnx.sum_weight_edges(edges, graph)
    ri_1 = internal_interconnectivity(graph, cid1), 
    ri_2 = internal_interconnectivity(graph, cid2)
    return ri_all / ((ri_1 + ri_2) / 2.0)

def relative_closeness(graph, cid1, cid2):
    partitions = (gnx.get_cluster_nodes(graph, cid1), gnx.get_cluster_nodes(graph, cid2))
    edges = gnx.connection_edges_between(partitions, graph)
    if len(edges) == 0:
        return 0.0
    rc_all = gnx.average_weight_edges(edges, graph)
    ic_1 = internal_closeness(graph, cid1)
    ic_2 = internal_closeness(graph, cid2)
    rc_1 = gnx.average_weight_edges(gnx.cluster_bisection_edges(graph, cid1), graph)
    rc_2 = gnx.average_weight_edges(gnx.cluster_bisection_edges(graph, cid2), graph)
    return rc_all / ((ic_1 / (ic_1 + ic_2) * rc_1) + (ic_2 / (ic_1 + ic_2) * rc_2))

def cost_func(graph, cid1, cid2, alpha):
    return relative_closeness(graph, cid1, cid2)**alpha * relative_interconnectivity(graph, cid1, cid2)

if __name__ == "__main__":
    rd = RawData(RawDataConfig(from_file = "data/data_01.pickle"))
    g = gnx.create_graphs(rd, 4)
    c_names, g = gnx.partition(g, 7)

    print(cost_func(g, 0, 1, alpha=1)) 
    print(cost_func(g, 1, 2, alpha=1)) 
    print(cost_func(g, 0, 2, alpha=1)) 
    visualise_2d_networkx(g, "cost_func.png", show_weight=True, color_clusters=True)