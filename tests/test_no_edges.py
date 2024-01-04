import src.graphs_igraph as gig
import src.graphs_networx as gnx
import pytest
from src.generate_data import *
from src.visualizers import * 

"""
Every cluster should be bisectable in a way which deletes some edges from 
graph, which can be used for further calculations in CHAMELEON.
"""

@pytest.mark.skip # graph cretion is unstable, it will lose 
def test_igraph():
    for i in range(100):
        rd = RawData(RawDataConfig(4,100,2, cluster_position_randomness=True))
        X = rd.data
        
        g = gig.create_graphs(rd, 20)
        names_clusters, g = gig.partition(g,3)

        sg = gig.get_cluster_subgraph(g,names_clusters[0])
        bisection = gig.bisect(sg)
        c_edges = gig.connection_edges_between(bisection, g)

        if len(c_edges) == 0:
            rd.save_data(f"data/alarmgig_{i}.pickle")
            assert False

def test_networx():
    for i in range(100):
        rd = RawData(RawDataConfig(4,100,2, cluster_position_randomness=True))
        X = rd.data
        
        g = gnx.create_graphs(rd, 20)
        names_clusters, g = gnx.partition(g,3)

        sg = gnx.get_cluster_subgraph(g,names_clusters[0])
        bisection = gnx.bisect(sg)
        c_edges = gnx.connection_edges_between(bisection, g)

        if len(c_edges) == 0:
            rd.save_data(f"data/alarmgnx_{i}.pickle")
            assert False

