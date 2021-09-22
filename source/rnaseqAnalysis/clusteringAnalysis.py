# %%
import numpy as np
import pandas as pd
from settings import params, paths
from scipy.io import mmread
import os
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from kneed import KneeLocator
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
'''
# Gene outlier removal (extreme max read counts > 2 * non-zero 99th percentile)
perc99 = np.percentile(countsRLE[countsRLE.nonzero()].ravel(), 99)

nonOutliers = np.max(countsRLE, axis=0) < perc99*10
countsRLE = countsRLE[:,  nonOutliers]
plt.figure()
plt.hist(countsRLE.ravel(), 20)
plt.yscale("log")
plt.xlabel("RLE counts")
plt.show()
'''
# %%
# Log transform, scale to unit variance and zero mean
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
countsScaled = np.log10(1+countsRLE)
countsScaled = scaler.fit_transform(countsScaled)
plt.figure()
plt.hist(countsScaled.ravel(), 20)
plt.yscale("log")
plt.xlabel("Z-scores")
plt.show()
# %%
# Outliers :/
plt.figure(dpi=500)
plt.boxplot(countsScaled[:100].T,showfliers=False)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()
# %%
# Remove experiments with abnormal median Z-score
IQRs = np.percentile(countsScaled, 75, axis=1)-np.percentile(countsScaled, 25, axis=1)
IQRmed = np.median(IQRs)
medians = np.median(countsScaled, axis=1)
nonOutliers = IQRmed/2 > np.abs(medians-np.median(medians))
countsScaled = countsScaled[nonOutliers]
plt.figure(dpi=500)
plt.boxplot(countsScaled[:100].T,showfliers=False)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()

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
