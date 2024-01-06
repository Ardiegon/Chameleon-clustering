import itertools

from tqdm import tqdm

import src.graphs_networx as gnx
import src.metrics as mtx
from src.visualizers import visualise_2d_networkx, create_gif

def merge(graph, cluster_names, target_cluster_number, alpha):
    max_score = 0
    best_ids = ()
    if len(cluster_names) <= target_cluster_number:
        return [], False

    for cid1, cid2 in itertools.combinations(cluster_names, 2):
        if cid1 != cid2:
            if not gnx.check_if_connected(graph, cid1, cid2):
                continue
            ms = mtx.cost_func(graph, cid1, cid2, alpha)
            if ms > max_score:
                max_score = ms
                best_ids = (cid1, cid2)

    if max_score > 0:
        for p in graph.nodes():
            if graph.nodes[p]["cluster_id"] == best_ids[0]:
                graph.nodes[p]["cluster_id"] = best_ids[1]
        return [best_ids[0]], True
    return [], False


def chameleon(raw_data, target_cluster_number, nearest_neighbors=10, minimum_cluster_nodes=7, alpha=0.5, plot = False):
    print("Building graph...")
    graph = gnx.create_graphs(raw_data, nearest_neighbors)
    c_names, graph = gnx.partition(graph, minimum_cluster_nodes)
    iters = len(c_names) - target_cluster_number
    
    if plot:    
        visualise_2d_networkx(graph, f"plots/chameleon_iter000.png", show_weight=False, color_clusters=True, show_node_ids=False)
    
    print("Clustering...")
    for i in tqdm(range(iters), total=iters):
        deleted_names, status = merge(graph, c_names, target_cluster_number, alpha)
        c_names = [cn for cn in c_names if cn not in deleted_names]
        if plot:
            idx = str(i+1).zfill(3)
            visualise_2d_networkx(graph, f"plots/chameleon_iter{idx}.png", show_weight=False, color_clusters=True, show_node_ids=False)
        if not status:
            break

    if plot:
        create_gif("plots", "plots/animation.gif")
    labels = []
    positions = []
    for n in graph.nodes():
        positions.append(graph.nodes[n]["pos"])
        labels.append(graph.nodes[n]["cluster_id"])
    mapping = {v: k for k, v in enumerate(list(set(labels)))}
    return positions, [mapping[v] for v in labels]

