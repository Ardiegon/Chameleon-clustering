import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import partition_igraph as pig
import igraph.drawing.colors as colors

from igraph import plot
from sklearn.neighbors import NearestNeighbors
from generate_data import RawData, RawDataConfig
from visualizers import visualise_2d_graph, visualise_clusters, visualise_hyperplane

try:
    import metis
    METIS_BACKEND = True
except:
    METIS_BACKEND = False
    print(
        """
        Warning! It's highly recommended to use Metis library, as said in original CHAMELEON paper.
        If you see this message, you probaby didn't installed pip metis wrapper, or its backend.
        If metis won't be installed, igraph fastgreedy will be used, which can cause errors in graphs 
        with lots of single unconnected vertices. Please configure Metis as showed here: 
            https://metis.readthedocs.io/en/latest/
        """
        )

DISTANCE_RESOLUTION = 1000

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
    graph["wadjlist"] = weighted_adjlist(graph)
    return graph

def weighted_adjlist(graph):
    adjacency_list_with_weights = []
    for vertex in graph.vs:
        neighbors_with_weights = []
        for neighbor in graph.neighbors(vertex):
            neighbors_with_weights.append((neighbor, int(graph.es[graph.get_eid(vertex.index, neighbor)]['weight']*DISTANCE_RESOLUTION)))
        adjacency_list_with_weights.append(neighbors_with_weights)
    return adjacency_list_with_weights

def partition(graph, min_cluster_indices = 3):
    n_vertices = graph.vcount()
    n_clusters = int(n_vertices / min_cluster_indices + 0.5)
    
    adjlist = graph["wadjlist"]
    _, parts = metis.part_graph(adjlist, n_clusters)
    graph.vs["cluster_id"] = parts
    return list(set(parts)), graph

# def partition(graph, min_cluster_indices = 3):
#     n_vertices = graph.vcount()
#     k  = int(n_vertices / min_cluster_indices + 0.5)
#     print(k)

#     clusters = 0
#     for i, p in enumerate(graph.vs):
#         graph.vs[p.index]['cluster_id'] = 0
#     cnts = {}
#     cnts[0] = n_vertices

#     while clusters < k - 1:
#         maxc = -1
#         maxcnt = 0
#         for key, val in cnts.items():
#             if val > maxcnt:
#                 maxcnt = val
#                 maxc = key
#         s_nodes = [n for n in graph.vs if graph.vs[n.index]['cluster_id'] == maxc]
#         s_graph = graph.subgraph(s_nodes)
#         map_s_indexes = {v.index: v["index"] for v in s_graph.vs}
#         s_weighted_adjlist = weighted_adjlist(s_graph)
#         edgecuts, parts = metis.part_graph(
#             s_weighted_adjlist, 2, objtype='cut', ufactor=250)
#         new_part_cnt = 0
#         for i, p in enumerate(s_graph.vs["index"]):
#             if parts[i] == 1:
#                 graph.vs[p]['cluster_id'] = clusters + 1
#                 new_part_cnt = new_part_cnt + 1
#         cnts[maxc] = cnts[maxc] - new_part_cnt
#         cnts[clusters + 1] = new_part_cnt
#         clusters = clusters + 1

#     return clusters, graph

def bisect(subgraph):
    adjlist = subgraph["wadjlist"]
    _, parts = metis.part_graph(adjlist, 2)
    ans = ([],[])
    for i in range(len(parts)):
        ans[parts[i]].append(subgraph.vs[i]["index"])
    return ans    
    
def connection_edges_between(bisection, graph):
    connection_edges = []
    for ver1 in bisection[0]:
        for ver2 in bisection[1]:
                if ver2 in graph.neighbors(ver1):
                    connection_edges.append((ver1, ver2))
    return connection_edges

def get_cluster_vertices_ids(graph, cluster_id):
    return [v["index"] for v in graph.vs if v["cluster_id"]==cluster_id]
    
def get_cluster_subgraph(graph, cluster_id):
    subgraph = graph.induced_subgraph(get_cluster_vertices_ids(graph,cluster_id))
    subgraph["wadjlist"] = weighted_adjlist(subgraph)
    return subgraph

def get_edges_average_weight(edges_list, graph):
    return sum([graph.es[graph.get_eid(e[0], e[1])]['weight'] for e in edges_list])/len(edges_list)


if __name__ == "__main__":
    rd = RawData(RawDataConfig(from_file = "data/data_01.pickle"))
    # rd = RawData(RawDataConfig(4,20,2, cluster_position_randomness=True))
    X = rd.data
    
    dist, ind = knn(X, 7)
    g = create_graphs(dist, ind)

    cluster_names, g = partition(g,3)
    print(len(g.vs["cluster_id"]))
    print(g.vs["cluster_id"])
    print(get_cluster_vertices_ids(g,cluster_names[0]))
    sg = get_cluster_subgraph(g,cluster_names[0])
    
    bisection = bisect(sg)
    print(bisection)
    c_edges = connection_edges_between(bisection, g)
    print(c_edges)

# VISUALIZATION
visualise_hyperplane(rd, [0,1], "here.png")
visualise_2d_graph(g, "knn_graph.png", show_weight=False)
visualise_2d_graph(sg, "knn_subgraph.png", show_weight=True)
visualise_clusters(g, "clusters.png", vis_dimension=[0,1], cluster_names=cluster_names)
visualise_clusters(sg, "subclusters.png", vis_dimension=[0,1], cluster_names=[cluster_names[0]])

# OTHER DATA 
# rd = RawData(RawDataConfig(from_file = "data/data_01.pickle"))
# rd = RawData(RawDataConfig(4,100,20, cluster_position_randomness=True))
# rd = RawData(RawDataConfig(4,5,2, cluster_position_randomness=True))
# rd.save_data("data/data_03.pickle")