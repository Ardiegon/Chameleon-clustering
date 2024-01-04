import src.graphs_igraph as gig
import src.graphs_networx as gnx
import pytest
from src.generate_data import *
from src.visualizers import * 


def test_graph_backbone_networx():
    rd = RawData(RawDataConfig(from_file = "data/data_03.pickle"))
    g = gnx.create_graphs(rd, 10)
    _, g = gnx.partition(g, 10)
    sg = gnx.get_cluster_subgraph(g, 0)
    partitions = gnx.bisect(sg)
    c_edges = gnx.connection_edges_between(partitions, g)
    assert c_edges==[(1025, 993), (1025, 962), (1025, 1157), (1025, 1160)]