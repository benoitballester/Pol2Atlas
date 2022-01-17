import numpy as np
import pynndescent
import igraph
import leidenalg
from sklearn.metrics import balanced_accuracy_score
from kneed import KneeLocator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import MiniBatchKMeans
from fastcluster import linkage_vector
import scipy.cluster.hierarchy as hierarchy

def graphClustering(matrix, metric, k="auto", r=0.4, snn=True, disconnection_distance=None, restarts=1):
    """
    Performs graph based clustering on the matrix.

    Parameters
    ----------
    metric: "auto" or string (optional, default "auto")
        Metric used for nn query. If set to "auto", it will use the jaccard index 
        on the experiments and the hamming distance on the consensuses
        if the peak scoring is set to binary, or pearson correlation otherwise. 
        See the pynndescent documentation for a list of available metrics.

    r: float (optional, default 0.4)
        Resolution parameter of the graph partitionning algorithm. Lower values = less clusters.

    k: "auto" or integer (optional, default "auto")
        Number of nearest neighbors used to build the NN graph.
        If set to auto uses 2*numPoints^0.2 neighbors as a rule of thumb, as too few 
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
    if k == "auto":
        k = int(np.power(len(matrix), 0.2)*2)
    # Add a few extra NNs to compute in order to get more accurate ANNs
    extraNN = 10
    lowMem = len(matrix) > 100000
    index = pynndescent.NNDescent(matrix, n_neighbors=k+extraNN+1, metric=metric, 
                                 low_memory=lowMem, random_state=42)
    nnGraph = index.neighbor_graph[0][:, 1:k+1]
    dists = index.neighbor_graph[1][:, 1:k+1]
    edges = np.zeros((nnGraph.shape[0]*nnGraph.shape[1], 2), dtype='int64')
    if snn:
        weights = np.zeros((nnGraph.shape[0]*nnGraph.shape[1]), dtype='float')
    for i in range(len(nnGraph)):
        for j in range(nnGraph.shape[1]):
            if nnGraph[i, j] > -0.5:    # Pynndescent may fail to find nearest neighbors in some cases
                if disconnection_distance is not None:
                    if dists[i, j] >= disconnection_distance:
                        continue
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

def autoRankPCA(mat, whiten=True, plot=True, minRank=0, maxRank=None):
    '''
    Performs PCA and select the adequate PCA dimensionnality using the elbow
    method.

    Parameters
    ----------
    mat: ndarray
        Input matrix

    whiten: boolean (optional, default False)
        Whether to whiten the PCA space or not
    
    plot: boolean (optional, default True)
        Plots the elbow curve

    maxRank: None or integer (optional, default None)
        Maximum PCA rank to compute. If set to None, selects the 
        smallest dimension of the matrix.

    Returns
    -------
    decomp : ndarray
        PCA space with optimal rank according to the elbow method.
    '''
    if maxRank is None:
        maxRank = np.min(mat.shape)
    model = PCA(maxRank, whiten=whiten)
    decomp = model.fit_transform(mat)
    kneedl = KneeLocator(np.arange(maxRank), model.explained_variance_, 
                         direction="decreasing", curve="convex", online=True)
    bestR = kneedl.knee
    if plot:
        plt.figure(dpi=300)
        kneedl.plot_knee()
        plt.xlabel("Principal component")
        plt.ylabel("Explained variance")
        plt.show()
    return decomp[:, :max(bestR,minRank)]

def threeStagesHC(matrix, metric, kMetaSamples=50000, method="ward"):
    """
    Three steps Hierachical clustering. UMAP -> K-Means -> Ward HC on clusters
    centroids.

    Parameters
    ----------
    matrix : array-like
        Data matrix
    
    metric : string
        Metric used for nn query. It is recommended to use Pearson correlation
        for float values and Dice similarity for binary data.
        See the pynndescent documentation for a list of available metrics.
    
    kMetaSamples : int, optional (default 50000)
        Number of K-Means clusters, or groups of samples used by HC.

    method : string, optional (default "ward")
        HC method
    """
    # First perform dimensionnality reduction
    lowMem = len(matrix) < 100000
    embedding = umap.UMAP(n_components=20, min_dist=0.0, n_neighbors=30, 
                          low_memory=lowMem, random_state=42, metric=metric).fit_transform(matrix)
    embedding = np.nan_to_num(embedding, nan=1e5)
    # Aggregrate samples via K-means in order to scale to large datasets
    if len(embedding) > kMetaSamples:
        clustering = MiniBatchKMeans(n_clusters=kMetaSamples, init="random", random_state=42, 
                                    n_init=1)
        assignedClusters = clustering.fit_predict(embedding)
        Kx = clustering.cluster_centers_
        # Perform HC
        link = linkage_vector(Kx, method=method)
        Korder = hierarchy.leaves_list(link)
        order = np.array([], dtype="int")
        for c in Korder:
            order = np.append(order, np.where(c == assignedClusters)[0])
        return order
    else:
        link = linkage_vector(embedding, method=method)
        return hierarchy.leaves_list(link)

def HcOrder(mat, method="ward", metric="euclidean"):
    link = linkage_vector(mat, method=method, metric=metric)
    rowOrder = hierarchy.leaves_list(link)
    link = linkage_vector(mat.T, method=method, metric=metric)
    colOrder = hierarchy.leaves_list(link)
    return rowOrder, colOrder

def looKnnCV(X, Y, metric, k):
    index = pynndescent.NNDescent(X, n_neighbors=min(30+k, len(X)-1), 
                                metric=metric, low_memory=False, random_state=42)
    # Exclude itself and select NNs (equivalent to leave-one-out cross-validation)
    # Pynndescent is ran with a few extra neighbors for a better accuracy on ANNs
    nnGraph = index.neighbor_graph[0][:, 1:k+1]
    pred = []
    for nns in Y[nnGraph]:
        # Find the most represented annotation in the k nearest neighbors
        pred.append(np.argmax(np.bincount(nns)))
    score = balanced_accuracy_score(Y, pred)
    return balanced_accuracy_score(Y, pred)
