import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import partition_igraph as pig
import igraph.drawing.colors as colors
from igraph import plot
from sklearn.neighbors import NearestNeighbors
from generate_data import RawData, RawDataConfig
from visualizers import visualise_2d_graph, visualise_clusters

def knn(X, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances, indices

def create_graphs(distances, indices):
    n_vertices = len(indices)
    n_neighbors = len(indices[0]-1)
    
    edges = []
    for index in indices:
        for i in range(1, n_neighbors):
            edges.append(tuple(sorted([index[0], index[i]])))
    edges = set(edges)

    weights = []
    for edge in edges:
        try:
            weights.append(distances[edge[0]][list(indices[edge[0]]).index(edge[1])])
        except:
            weights.append(distances[edge[1]][list(indices[edge[1]]).index(edge[0])])

    graph = ig.Graph(n_vertices, edges)
    graph.vs["index"] = [i for i in range(n_vertices)]
    graph.vs["X"] = X
    graph.es["weight"] = weights
    return graph

def graph_partition(graph, max_cluster_indices = 3):
    n_vertices = graph.vcount()
    n_clusters = int(n_vertices / max_cluster_indices + 0.5) 
    clusters = g.community_fastgreedy(g.es["weight"]).as_clustering(n_clusters)
    print(clusters.membership)
    graph.vs["cluster_id"] = clusters.membership
    return graph

def graph_connection(g1, g2):
    pass


if __name__ == "__main__":
    # X = RawData(RawDataConfig(4,100,2, cluster_position_randomness=True)).data
    # X = RawData(RawDataConfig(4,5,2, cluster_position_randomness=True)).data
    X = RawData(RawDataConfig(from_file = "data\\data_01.pickle")).data
    dist, ind = knn(X, 4)
    g = create_graphs(dist, ind)
    g = graph_partition(g, 4)
    visualise_2d_graph(g, "knn_graph.png", show_weight=True)
    visualise_clusters(g, "clusters.png", vis_dimension=[0,1])
