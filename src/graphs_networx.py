import metis
import networkx as nx
import numpy as np

from sklearn.neighbors import NearestNeighbors

from src.generate_data import RawData, RawDataConfig
from src.visualizers import  visualise_clusters, visualise_hyperplane, visualise_2d_networkx

def knn(X, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances, indices

def create_graphs(raw_data, n_neighbors):
    distances, indices = knn(raw_data.data, n_neighbors)

    n_neighbors = len(indices[0]-1)
    
    edges = []
    positions = {}
    for pos, index in enumerate(indices):
        positions[pos] = raw_data.data[pos]
        for i in range(1, n_neighbors):
            edges.append(tuple(sorted([index[0], index[i]])))
    edges = set(edges)

    weights = []
    for edge in edges:
        try:
            weights.append(distances[edge[0]][list(indices[edge[0]]).index(edge[1])])
        except:
            weights.append(distances[edge[1]][list(indices[edge[1]]).index(edge[0])])

    graph = nx.Graph()

    for edge, weight in zip(edges, weights):
        graph.add_edge(edge[0], edge[1], weight=weight)

    nx.set_node_attributes(graph, positions, 'pos')
    
    return graph

def partition(graph, min_cluster_indices = 3):
    def merge_to_closest_connected(subgraph, graph):
        p_edges = []
        p_weights = []
        for n in subgraph.nodes:
            for m in graph[n]:
                if m in subgraph.nodes:
                    continue
                weight = graph.get_edge_data(n, m).get('weight')
                p_edges.append((n, m))
                p_weights.append(weight)
        best_edge = p_edges[p_weights.index(min(p_weights))]
        deleted_cluster = graph.nodes[best_edge[0]]["cluster_id"]
        closest_cluster = graph.nodes[best_edge[1]]["cluster_id"]
        for n in subgraph.nodes:
            graph.nodes[n]["cluster_id"] = closest_cluster
        return deleted_cluster, closest_cluster
    
    def rename_clusters(graph, c_names):
        for id, name in enumerate(sorted(c_names)):
            if id == name:
                continue
            for n in graph.nodes:
                if graph.nodes[n]["cluster_id"] == name:
                    graph.nodes[n]["cluster_id"] = id 
        new_c_names = nx.get_node_attributes(graph, 'cluster_id').values()
        return sorted(list(set(new_c_names)))

    n_vertices = graph.number_of_nodes()
    n_clusters = int(n_vertices / min_cluster_indices + 0.5)
    
    _, parts = metis.part_graph(graph, n_clusters) # contig = True throws metis.METIS_OtherError
    for n, c_id in zip(graph.nodes, parts):
        graph.nodes[n]["cluster_id"] = c_id

    c_names = list(set(parts))
    next_cluster =  max(c_names) + 1
    found_unconnected_cluster = True
    while found_unconnected_cluster:
        found_unconnected_cluster = False
        c_names_buff = []
        merged_clusters = []
        for cid in c_names:
            subgraph = get_cluster_subgraph(graph, cid)
            partitions = bisect(subgraph)
            c_edges = connection_edges_between(partitions, graph)
            if len(c_edges) == 0:
                found_unconnected_cluster = True
                if len(partitions[0]) == 0 or len(partitions[1])==0:
                    c_deleted, c_merged = merge_to_closest_connected(subgraph, graph)
                    merged_clusters.append(c_deleted)
                else:
                    for n in partitions[1]:
                        graph.nodes[n]['cluster_id'] = next_cluster
                    c_names_buff.append(next_cluster)
                    next_cluster+=1
        c_names += c_names_buff
        c_names = [cn for cn in c_names if cn not in merged_clusters]

    c_names = rename_clusters(graph, c_names)
    return c_names, graph

def bisect(subgraph):
    _, parts = metis.part_graph(subgraph, 2, ufactor=100)
    ans = ([],[])
    for i, n in enumerate(subgraph.nodes):
        ans[parts[i]].append(n)
    return ans    

def cluster_bisection_edges(graph, cluster_id):
    subgraph = get_cluster_subgraph(graph, cluster_id) 
    partitions = bisect(subgraph=subgraph)
    return connection_edges_between(partitions, graph)
    
def connection_edges_between(bisection, graph):
    connection_edges = []
    for ver1 in bisection[0]:
        for ver2 in bisection[1]:
                if ver2 in graph[ver1]:
                    connection_edges.append((ver1, ver2))
    return connection_edges

def get_cluster_subgraph(graph, cluster_id):
    nodes = [n for n in graph.nodes if graph.nodes[n]['cluster_id']==cluster_id]
    subgraph = graph.subgraph(nodes)
    return subgraph

def get_two_clusters_subgraph(graph, cid1, cid2):
    nodes = [n for n in graph.nodes if graph.nodes[n]['cluster_id']==cid1 or graph.nodes[n]['cluster_id']==cid2]
    subgraph = graph.subgraph(nodes)
    return subgraph

def get_cluster_nodes(graph, cluster_id):
    return [n for n in graph.nodes if graph.nodes[n]['cluster_id']==cluster_id]

def sum_weight_edges(edge_list, graph, return_count = False):
    total_weight = 0
    count = 0

    for edge in edge_list:
        node_id_1, node_id_2 = edge
        if graph.has_edge(node_id_1, node_id_2):
            total_weight += graph[node_id_1][node_id_2].get('weight', 0)
            count += 1

    if return_count:
        return count, total_weight
    else: 
        return total_weight


def average_weight_edges(edge_list, graph):
    count, total_weight = sum_weight_edges(edge_list, graph, return_count=True)

    if count == 0:
        return 0

    average_weight = total_weight / count
    return average_weight

if __name__ == "__main__":
    # rd = RawData(RawDataConfig(from_file = "data/data_01.pickle"))
    rd = RawData(RawDataConfig(from_file = "data/test_00.pickle"))
    # rd = RawData(RawDataConfig(4,50,10, cluster_position_randomness=True))

    g = create_graphs(rd, 10)

    c_names, g = partition(g, 10)
    print(c_names)

    sg = get_cluster_subgraph(g, 0)

    partitions = bisect(sg)
    print(partitions)

    c_edges = connection_edges_between(partitions, g)
    print(c_edges)

    count, s_edge_weight = sum_weight_edges(c_edges, g)
    a_edge_weight = average_weight_edges(c_edges,g)
    print(s_edge_weight)
    print(a_edge_weight)
    
    # VISUALIZATIONS
    visualise_hyperplane(rd, [0,1], "here.png")
    visualise_2d_networkx(g, "nx_graph.png")
    visualise_2d_networkx(g, "nx_graph_clusters.png", color_clusters=True)
    visualise_2d_networkx(sg, "nx_subgraph.png", color_clusters=False)




