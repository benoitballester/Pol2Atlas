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

# %%
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
dlFiles = os.listdir(paths.countDirectory + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
for f in dlFiles:
    try:
        id = f.split(".")[0]
        # countsBG.append(pd.read_csv(paths.countDirectory + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(paths.countDirectory + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(paths.countDirectory + "500centroid/" + f, header=None, skiprows=2).values)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except:
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
# bgCounts = np.concatenate(countsBG, axis=1).T
# %%
# Keep tumoral samples
kept = np.isin(order, annotation.index)
allCounts = allCounts[kept]
# bgCounts = bgCounts[:, kept]
allReads = allReads[kept]
annotation = annotation.loc[np.array(order)[kept]]
kept = np.logical_not(annotation["Sample Type"] == "Solid Tissue Normal")
annotation = annotation[kept]
allCounts = allCounts[kept]
# bgCounts = bgCounts[kept]
allReads = allReads[kept]
# %%
# Remove undected Pol II probes
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=3)
counts = allCounts[:, nzCounts] 
# Convert reads to ranks
ranks = rankdata(counts, axis=1)
# Rescale ranks to unit Variance for numerical stability (assuming uniform distribution for ranks)
rgs = (ranks / ranks.shape[1] - 0.5) * np.sqrt(12)
# %%
# Feature selection based on rank variability
selected = rnaseqFuncs.variableSelection(rgs, plot=True)
# %%
import umap
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                      metric="euclidean").fit_transform(decomp)
# %%
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch

tcgaProjects = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/tcga_project_annot.csv", 
                            sep="\t", index_col=0)
project_id = annotation["project_id"]
# cancerType, eq = pd.factorize(tcgaProjects["Origin"].loc[project_id])
cancerType, eq = pd.factorize(annotation["project_id"])
plt.figure(dpi=500)
palette, colors = getPalette(cancerType)
# allReadsScaled = (allReads - allReads.min()) / (allReads.max()-allReads.min())
# col = sns.color_palette("rocket_r", as_cmap=True)(allReadsScaled)
plt.scatter(embedding[:, 0], embedding[:, 1], s=min(10.0,100/np.sqrt(len(embedding))),
            linewidths=0.0, c=colors)
xScale = plt.xlim()[1] - plt.xlim()[0]
yScale = plt.ylim()[1] - plt.ylim()[0]
# plt.gca().set_aspect(xScale/yScale)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/global/umap_all_tumors.png")
plt.show()
plt.figure(dpi=500)
plt.axis('off')
patches = []
for i in np.unique(cancerType):
    legend = Patch(color=palette[i], label=eq[i])
    patches.append(legend)
plt.legend(handles=patches)
plt.savefig(paths.outputDir + "rnaseq/global/umap_all_tumors_lgd.png", bbox_inches="tight")
plt.show()
# %%
orderRows = matrix_utils.threeStagesHC(rgs[:, selected], "correlation")
orderCols = matrix_utils.threeStagesHC(rgs[:, selected].T, "correlation")
# %%
plot_utils.plotHC((rgs/np.max(rgs)).T, project_id, None, rowOrder=orderRows, colOrder=orderCols)

# %%
print("Rank + selection", matrix_utils.looKnnCV(rgs[:, selected], 
                            cancerType, "correlation", 1))
print("Rank", matrix_utils.looKnnCV(rgs, 
                            cancerType, "correlation", 1))
# %%
from scipy.special import erfinv
fpkmCounts = counts / allReads[:, None]
rgPerGene = (rankdata(fpkmCounts, axis=0)-0.5)/fpkmCounts.shape[0]*2.0-1.0
quantileNormed = erfinv(rgPerGene)
# %%
print("log fpkm", matrix_utils.looKnnCV(np.log(1 + 1e4 * counts / np.sum(counts, axis=1)[:, None])[:, selected], 
                            cancerType, "correlation", 30))
selectedFpkm = rnaseqFuncs.variableSelection(fpkmCounts, plot=True)
print("log fpkm + selection", matrix_utils.looKnnCV(rgs[:, selected],
                            cancerType, "correlation", 30))

# %%
decomp = matrix_utils.autoRankPCA(rgs[:, selected])
print("log fpkm + selection", matrix_utils.looKnnCV(decomp,
                            cancerType, "correlation", 1))
# %%
