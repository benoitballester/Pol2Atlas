# %%
import numpy as np
import pandas as pd
from settings import params, paths
from scipy.io import mmread
import os
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from kneed import KneeLocator
from settings import params, paths
from lib import normRNAseq
from lib.utils import plot_utils, matrix_utils
from scipy.special import expit
from sklearn.preprocessing import StandardScaler, RobustScaler, power_transform

countDir = "/scratch/pdelangen/projet_these/outputPol2/rnaseq/encode_counts/"
# %%
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/encode_total_rnaseq_annot_0.tsv", 
                        sep="\t", index_col=0)
dlFiles = os.listdir(countDir + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
for f in dlFiles:
    try:
        id = f.split(".")[0]
        # countsBG.append(pd.read_csv(paths.countDirectory + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except:
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
ann, eq = pd.factorize(annotation.loc[order]["Annotation"])
# %%
nzCounts = np.sum(allCounts >= 1, axis=0) >=3
counts = allCounts[:, nzCounts] 
# Remove outliers with extreme read counts (usually ribosomes)
outlierCounts = np.percentile(counts, 99, 0) < np.percentile(np.max(counts, axis=0), 99)
counts = counts[:, outlierCounts]
# %%
# Deviance based feature selection
from kneed import KneeLocator
n_i = np.sum(counts, axis=1)[:, None]
countsProp = counts/n_i
pi_j = np.mean(countsProp, axis=0)
v = counts * np.log(1e-15 + counts / (n_i*pi_j)) + (n_i - counts) * np.log(1e-15 + (n_i - counts)/(n_i * (1-pi_j)))
deviance = np.sum(v, axis=0)
# %%
orderedDev = np.argsort(deviance)[::-1]
kneedl = KneeLocator(np.arange(len(deviance)), deviance[orderedDev],
                     direction="decreasing", curve="convex", online=True)
bestR = kneedl.knee
kneedl.plot_knee()
selected = orderedDev[:bestR]
bestR
# %%
import umap
from scipy.stats import rankdata
rgs = rankdata(counts[:, orderedDev[:bestR]], axis=1)
embedding = umap.UMAP(n_neighbors=50, min_dist=0.5, random_state=0, low_memory=False, metric="correlation").fit_transform(rgs)
plt.figure(figsize=(10,10), dpi=500)
palette, colors = plot_utils.getPalette(ann)
plot_utils.plotUmap(embedding, colors)
patches = []
for i in np.unique(ann):
    legend = Patch(color=palette[i], label=eq[i])
    patches.append(legend)
plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
plt.show()
# %%
import seaborn as sns
from scipy.stats import rankdata
# Mean-variance relationship
logCounts = np.log2(1+counts)
mc = np.mean(logCounts, axis=0)
vc = np.var(logCounts, axis=0)
pct = rankdata(deviance)/len(deviance)
color = sns.color_palette("magma", as_cmap=True)(pct)
plt.figure(dpi=500)
plt.scatter(mc, vc, c=color, s=0.2, linewidths=0.0)
plt.gca().set_facecolor((0.5,0.5,0.5))
plt.show()
plt.figure(dpi=500)
color = np.array([[0,0,1]]*len(mc))
color[orderedDev[:bestR]] = [1,0,0]
plt.scatter(mc, vc, c=color, s=0.2, linewidths=0.0)
plt.gca().set_facecolor((0.5,0.5,0.5))
plt.show()
# %%
clustsPol2 = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt").astype(int)
nClusts = np.max(clustsPol2)+1
nAnnots = np.max(ann)+1
ranksums = np.zeros((nClusts, nAnnots))
for i in range(np.max(clustsPol2)+1):
    pass