import numpy as np
import pynndescent
import igraph
import leidenalg


def graphClustering(matrix, metric, k="auto", r=0.4, snn=True, restarts=1):
    """
    Performs graph based clustering on the matrix.

    Parameters
    ----------
    metric: "auto" or string (optional, default "auto")
        Metric used by umap. If set to "auto", it will use the jaccard index 
        on the experiments and the hamming distance on the consensuses
        if the peak scoring is set to binary, or pearson correlation otherwise. 
        See the UMAP documentation for a list of available metrics.

    r: float (optional, default 0.4)
        Resolution parameter of the graph partitionning algorithm. Lower values = less clusters.

    k: "auto" or integer (optional, default "auto")
        Number of nearest neighbors used to build the NN graph.
        If set to auto uses numPoints^0.33 neighbors as a rule of thumb, as too few 
        NN with a lot of points can create disconnections in the graph.

    snn: Boolean (optional, default True)
        If set to True, it will perform the Shared Nearest Neighbor Graph 
        clustering variant, where the edges of the graph are weighted according 
        to the number of shared nearest neighbors between two nodes. Otherwise,
        all edges are equally weighted. SNN can produce a more refined clustering 
        but it can also hallucinate some clusters.

    restarts: integer (optional, default 1)
        The number of times to restart the graph partitionning algorithm, before keeping 
        the best partition according to the quality function.
    
    Returns
    -------
    labels: ndarray
        Index of the cluster each sample belongs to.
    """
    # Create NN graph
    k = int(np.power(len(matrix), 0.33) + 1)
    index = pynndescent.NNDescent(matrix, n_neighbors=k+1, metric=metric, 
                                  diversify_prob=0.0, pruning_degree_multiplier=9.0)
    nnGraph = index.neighbor_graph[0][:, 1:]
    edges = np.empty((nnGraph.shape[0]*nnGraph.shape[1], 2), dtype='int64')
    if snn:
        weights = np.empty((nnGraph.shape[0]*nnGraph.shape[1]), dtype='float')
    for i in range(len(nnGraph)):
        for j in range(nnGraph.shape[1]):
            if nnGraph[i, j] > -0.5:    # Pynndescent may fail to find nearest neighbors in some cases
                link = nnGraph[i, j]
                edges[i*nnGraph.shape[1]+j] = [i, link]
                if snn:
                    # Weight the edges based on the number of shared nearest neighbors between two nodes
                    weights[i*nnGraph.shape[1]+j] = len(np.intersect1d(nnGraph[i], nnGraph[link]))
    graph = igraph.Graph(n=len(nnGraph), edges=edges, directed=True)
    # Restart clustering multiple times and keep the best partition
    best = -np.inf
    partitions = None
    for i in range(restarts):
        if snn:
            part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, 
                                            seed=i, resolution_parameter=r, weights=weights, n_iterations=-1)
        else:
            part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, 
                                            seed=i,resolution_parameter=r, n_iterations=-1)
        if part.quality() > best:
            partitions = part
            best = part.quality()
    # Map partitions to per object assignments
    clustered = np.zeros(len(nnGraph), dtype="int")
    for i, p in enumerate(partitions):
        clustered[p] = i
    return clustered