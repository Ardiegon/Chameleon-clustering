import matplotlib.pyplot as plt
import igraph as ig

from sklearn.neighbors import NearestNeighbors
from generate_data import RawData, RawDataConfig
from visualizers import visualise_2d_graph


def create_graphs_knn(X, n_neighbors):
    n_vertices = len(X)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    edges = []
    for index in indices:
        for i in range(1, n_neighbors):
            edges.append(tuple(sorted([index[0], index[i]])))
    edges = set(edges)

    graph = ig.Graph(n_vertices, edges) 
    graph.vs["index"] = [i for i in range(n_vertices)]
    graph.vs["X"] = X
    return graph

def graph_partition(graph):
    pass

def graph_connection(g1, g2):
    pass


if __name__ == "__main__":
    X = RawData(RawDataConfig(4,100,2, cluster_position_randomness=True)).data
    g = create_graphs_knn(X, 4)
    visualise_2d_graph(g, "knn_graph.png")


