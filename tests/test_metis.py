import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import metis

def test_metis():
    G = nx.Graph()
    nodes = range(10)
    G.add_nodes_from(nodes)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]
    G.add_edges_from(edges)
    (edgecuts, parts) = metis.part_graph(G, 3)
    assert parts==[0, 1, 1, 1, 2, 2, 2, 0, 0, 0]

test_metis()
