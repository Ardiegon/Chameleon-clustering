import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualise_hyperplane(raw_data, vis_dimension, path, color_classes = True):
    X, y = raw_data.data, raw_data.labels
    assert len(vis_dimension)==2
    assert all(vd < len(X[0]) for vd in vis_dimension)

    n_classes = len(np.unique(y)) if y is not None else 0
    colors = cm.rainbow(np.linspace(0, 1, n_classes)) if color_classes else ["gray" for _ in range(n_classes)]

    for c in range(n_classes):
        plt.scatter(X[y == c, vis_dimension[0]], X[y == c, vis_dimension[1]], s=10, color=colors[c], label=f"Class {c}")
    plt.title(f"Hyperplane of dimensions {vis_dimension[0]} and {vis_dimension[1]}")
    plt.savefig(path)
    plt.cla()

def visualise_hypercube(raw_data, vis_dimension, path, color_classes = True):
    X, y = raw_data.data, raw_data.labels
    assert len(vis_dimension)==3
    assert all(vd < len(X[0]) for vd in vis_dimension)    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n_classes = len(np.unique(y)) if y is not None else 0
    colors = cm.rainbow(np.linspace(0, 1, n_classes)) if color_classes else ["gray" for _ in range(n_classes)]

    for c in range(n_classes):
        ax.scatter(X[y == c, vis_dimension[0]], X[y == c, vis_dimension[1]], X[y == 0, vis_dimension[2]], s=10, color=colors[c], label=f"Class {c}")
    plt.title(f"Hypercube of dimensions {vis_dimension[0]} and {vis_dimension[1]} and {vis_dimension[2]}")
    plt.savefig(path)
    plt.cla()

def visualise_2d_igraph(graph, path, show_weight = False):
    """
    Shows graph. If length of vector describing location 
    of single vertex is longer than 2, it still shows only 
    first and second dimension.
    """
    import igraph as ig
    fig, ax = plt.subplots(figsize=(5,5))
    n_dims = len(graph.vs["X"][0])
    ig.plot(
        graph, 
        target=ax,
        layout=graph.vs["X"] if n_dims <= 2 else np.array(graph.vs["X"])[:, :2],
        vertex_size=10,
        vertex_color="black",
        vertex_frame_width=0.5,
        vertex_frame_color="white",
        vertex_label = [v for v in graph.vs["index"]] if show_weight else ["" for _ in range(graph.vcount())],
        edge_label = [f'{w:.3f}' for w in graph.es["weight"]] if show_weight else ["" for _ in range(graph.vcount())],
        edge_align_label = True,
        edge_label_size = 6,
    )
    plt.savefig(path)
    plt.cla()

def visualise_2d_networkx(graph, path, show_weight=False, color_clusters=False):
    """
    Shows graph. If the length of the vector describing the location 
    of a single vertex is longer than 2, it still shows only 
    the first and second dimension.
    """
    pos = nx.get_node_attributes(graph, 'pos')
    if len(pos.values()[0]) > 2:
        for k, v in pos.items():
            pos[k] = v[:2]
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    for k, v in edge_labels.items():
        edge_labels[k] = f"{v:.2f}"

    fig, ax = plt.subplots()

    if color_clusters:
        node_colors = list(nx.get_node_attributes(graph, "cluster_id").values())
        cmap = plt.cm.rainbow
        norm = plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        nx.draw(graph, pos, with_labels=True, node_size=600, node_color=node_colors, cmap=cmap,
                font_size=10, font_color="black", font_weight="bold", width=2, ax=ax)

        cbar = plt.colorbar(sm, ticks=list(set(node_colors)), ax=ax)
        cbar.set_label('Cluster ID')

        unique_clusters = sorted(list(set(node_colors)))
        legend_labels = [f"Cluster {label}" for label in unique_clusters]
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(label)),
                                       markersize=10) for label in unique_clusters],
                   labels=legend_labels, loc='upper right')

    else:
        nx.draw(graph, pos, with_labels=True, node_size=600, font_size=10,
                font_color="black", font_weight="bold", width=2, ax=ax)

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.savefig(path)
    plt.cla()

def visualise_clusters(graph, path, vis_dimension, cluster_names, networx = True):
    assert len(vis_dimension)==2
    if networx:
        X, y = np.array(nx.get_node_attributes(graph, 'pos').values()), np.array(nx.get_node_attributes(graph, 'cluster_id').values())
    else:
        X, y = np.array(graph.vs["X"]), np.array(graph.vs["cluster_id"])

    n_classes = len(np.unique(y)) if y is not None else 0
    colors = cm.rainbow(np.linspace(0, 1, n_classes))

    for i, c in enumerate(cluster_names):
        plt.scatter(X[y == c, vis_dimension[0]], X[y == c, vis_dimension[1]], s=10, color=colors[i], label=f"Class {c}")
    plt.title(f"Clusters - visualised dimensions {vis_dimension[0]} and {vis_dimension[1]}")
    plt.savefig(path)
    plt.cla()
