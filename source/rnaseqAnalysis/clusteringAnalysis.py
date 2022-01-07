# %%
import numpy as np
import pandas as pd
from settings import params, paths
from scipy.io import mmread
import os
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from kneed import KneeLocator
counts = np.asarray(mmread(paths.outputDir + "rnaseq/" + "counts.mtx").todense())
annotation = pd.read_csv(paths.outputDir + "rnaseq/annotation.tsv", sep="\t")
# %%
# Keep only tumors with annotated survival
selectedLines = (annotation["State"] == "Primary Tumor") & (annotation["Dead"] > -0.5)
counts = counts[selectedLines]
# %%
# Remove low counts + RLE normalization
nzPos = np.sum(counts > np.sum(counts, axis=1)[:, None]/20000, axis=0) > 2
counts = counts[:, nzPos]
# RLE normalization
m = gmean(counts, axis=0)
c1 = np.where(m > 0.9, counts / m, np.nan)
scales = np.nanmedian(c1, axis=1)[:, None]
countsRLE = counts / scales
countsRLE = countsRLE / np.min(countsRLE[countsRLE.nonzero()])
# %%
# Read consensus table
consensuses = pd.read_csv(paths.outputDir + "rnaseq/consensuses.bed", sep="\t")
pct95 = np.percentile(countsRLE, 95, axis=0)
order = np.argsort(-pct95)
consensuses = consensuses[["Chr", "Start", "End"]].loc[nzPos]
consensuses[4] = np.argsort(order)
consensuses[5] = pct95.astype(int)
consensuses.iloc[order].to_csv("ranked_95pct.bed", sep="\t", header=None, index=None)

# %%

# %%
# PCA, determine optimal number of K using the elbow method
from lib.utils.matrix_utils import autoRankPCA
decomp = autoRankPCA(countsScaled, whiten=True)
decomp.shape
# %%
from lib.utils.matrix_utils import graphClustering
import matplotlib.pyplot as plt
labels = graphClustering(countsScaled, "correlation", restarts=100)
import kaplanmeier as km
outValues = []
for i in np.unique(labels):
    inCluster = (np.array(labels) == i).astype(int)
    outValues.append(km.fit(annotation["Time_to_event"][selectedLines][nonOutliers], 
                           annotation["Dead"][selectedLines][nonOutliers], inCluster))
for i, o in enumerate(outValues):
    print(f"Cluster {i} vs All")
    km.plot(o)
    plt.show()

# %%
import umap
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch
embedding = umap.UMAP(metric="correlation").fit_transform(countsScaled)
# %%
plt.figure(dpi=500)
palette, colors = getPalette(labels)
plotUmap(embedding, colors)
patches = []
for i in np.unique(labels):
    legend = Patch(color=palette[i], label=str(i))
    patches.append(legend)
plt.legend(handles=patches)
plt.show()
# %%
cancerType, eq = pd.factorize(annotation["Type"][selectedLines][nonOutliers])
plt.figure(dpi=500)
palette, colors = getPalette(cancerType)
plotUmap(embedding, colors)
plt.show()
plt.figure()
patches = []
for i in np.unique(cancerType):
    legend = Patch(color=palette[i], label=eq[i])
    patches.append(legend)
plt.legend(handles=patches)
plt.show()
# %%
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
normalized_mutual_info_score(cancerType, labels)

# %%
