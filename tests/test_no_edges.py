from src.graphs import *
from src.visualizers import * 

for i in range(1):
    rd = RawData(RawDataConfig(from_file = "data/data_01.pickle"))
    # rd = RawData(RawDataConfig(4,100,2, cluster_position_randomness=True))
    X = rd.data
    
    dist, ind = knn(X, 4)
    g = create_graphs(dist, ind)

    n_clusters, g = partition(g,3)
    print(n_clusters)
    sg = get_cluster_subgraph(g,0)
    bisection = bisect(sg)
    c_edges = connection_edges_between(bisection, g)
    if len(c_edges) == 0:
        print("ALARM!")
        rd.save_data(f"data/alarm_{i}.pickle")

# VISUALIZATION
visualise_hyperplane(rd, [0,1], "here.png")
visualise_2d_graph(g, "knn_graph.png", show_weight=False)
visualise_2d_graph(sg, "knn_subgraph.png", show_weight=True)
visualise_clusters(g, "clusters.png", vis_dimension=[0,1])
visualise_clusters(sg, "subclusters.png", vis_dimension=[0,1])

# OTHER DATA 
# rd = RawData(RawDataConfig(from_file = "data/data_01.pickle"))
# rd = RawData(RawDataConfig(4,100,20, cluster_position_randomness=True))
# rd = RawData(RawDataConfig(4,5,2, cluster_position_randomness=True))
# rd.save_data("data/data_03.pickle")