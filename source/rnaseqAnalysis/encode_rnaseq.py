# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib import rnaseqFuncs
from lib.utils import plot_utils, matrix_utils
from matplotlib.patches import Patch
from scipy.stats import rankdata, chi2
from scipy.stats import chi2
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection
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
# Remove undected Pol II probes
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=3)
counts = allCounts[:, nzCounts] 
# Convert reads to ranks
ranks = rankdata(counts, "min", axis=1)
# Rescale ranks to unit Variance for numerical stability (assuming uniform distribution for ranks)
rgs = (ranks / ranks.shape[1] - 0.5) * np.sqrt(12)
# %%
# Feature selection based on rank variability
selected = rnaseqFuncs.variableSelection(rgs)
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=50, min_dist=1.0,
                     random_state=42, low_memory=False, metric="correlation").fit_transform(rgs[:, selected])
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
clustsPol2 = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt").astype(int)[nzCounts]
countsProp = counts / np.sum(counts, axis=1)[:, None]
nClusts = np.max(clustsPol2) + 1
nAnnots = np.max(ann) + 1
ranksums = np.zeros((nClusts, nAnnots))
for i in range(nClusts):
    inClust = clustsPol2 == i
    for j in range(nAnnots):
        hasAnnot = ann == j
        ranksums[i,j] = np.mean(ranks[hasAnnot][:, inClust])
# %%
rankSumNorm = ranksums / np.sum(ranksums, axis=0)
rankSumNorm = rankSumNorm - np.min(rankSumNorm, axis=0)
rankSumNorm = rankSumNorm / np.max(rankSumNorm, axis=0)
rankSumNorm = rankSumNorm / np.max(rankSumNorm, axis=1)[:, None]
colorMat = sns.color_palette("magma", as_cmap=True)(rankSumNorm**2.2)
plt.figure(dpi=300)
plt.imshow(colorMat[:16])
plt.gca().set_aspect(1)
# %%
