from src.graphs import *
from src.visualizers import * 

for i in range(1):
    rd = RawData(RawDataConfig(4,100,2, cluster_position_randomness=True))
    X = rd.data
    
    g = create_graphs(rd, 20)
    names_clusters, g = partition(g,3)

    sg = get_cluster_subgraph(g,names_clusters[0])
    bisection = bisect(sg)
    c_edges = connection_edges_between(bisection, g)

    if len(c_edges) == 0:
        print("ALARM!")
        rd.save_data(f"data/alarm_{i}.pickle")
