# %%
from lib.peakMerge import peakMerger
from lib.utils import overlap_utils, matrix_utils
import numpy as np
from settings import params, paths

# %%
# First merge peaks and generate data matrix
merger = peakMerger(paths.genomeFile, outputPath=paths.outputDir)
merger.mergePeaks(paths.peaksFolder, inferCenter=params.inferCenter, 
                  minOverlap=params.minOverlap, fileFormat=params.fileFormat)
merger.writePeaks()
# %%
# UMAP of samples
merger.umap(transpose=True, annotationFile=paths.annotationFile, reDo=True)
# %%
# Clustering samples
merger.clusterize(transpose=True, restarts=100, annotationFile=paths.annotationFile)
# %%
# UMAP of consensus peaks, using Autoencoder
from lib.AE import AE
model = AE()
latent = model.fit_transform(merger.matrix)
merger.umap(transpose=False, altMatrix=latent, metric="euclidean",
            annotationFile=paths.annotationFile)
# %% 
# Clustering consensuses autoencoder representation
merger.clusterize(transpose=False, altMatrix=latent, restarts=5, annotationFile=paths.annotationFile)
# %%
# Intersect enrichments
# %%
clusters = matrix_utils.graphClustering(decomp, "euclidean")
# %%
palette, colors = plot_utils.getPalette(clusters)
plt.figure(dpi=500)
plot_utils.plotUmap(merger.embedding[0], colors)
# %%
merger.clustered[0] = clusters
merger.clusterize(transpose=False, annotationFile=paths.annotationFile, reDo=False)
# %%
mcaMtx = merger2.matrix / np.mean(merger2.matrix, axis=1)[:, None]-1
decomp = matrix_utils.autoRankPCA(mcaMtx, whiten=True)
# %%
merger2.umap(transpose=False, altMatrix=decomp, metric="euclidean",
            annotationFile=paths.annotationFile, reDo=False)
# %%
merger2.clusterize(transpose=False, altMatrix=decomp, metric="euclidean",
                  annotationFile=paths.annotationFile, reDo=True)
# %%
mcaMtxT = merger2.matrix.T / np.mean(merger2.matrix.T, axis=1)[:, None]-1
decomp = matrix_utils.autoRankPCA(mcaMtxT, whiten=True)
# %%
merger2.umap(transpose=True, altMatrix=decomp, metric="euclidean",
            annotationFile=paths.annotationFile, reDo=True)
# %%
