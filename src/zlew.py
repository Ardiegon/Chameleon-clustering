# def partition_robust(graph, min_cluster_indices = 3):
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