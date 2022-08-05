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
# bgCounts = np.concatenate(countsBG, axis=1)
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
selected = rnaseqFuncs.variableSelection(rgs)
# %%
import umap
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch
embedding = umap.UMAP(min_dist=0.5, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(rgs[:, selected])
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
import seaborn as sns
# Sample corr
corr = np.corrcoef(countsScaled)
plt.figure(dpi=500)
sns.clustermap(countsScaled, method="ward", row_colors=colors, col_cluster=True)
plt.show()
# %%
'''
from scipy.stats import mannwhitneyu, ranksums, ttest_ind
# Evaluating correlation of expression per cluster of pol II

clusters = pd.read_csv(paths.outputDir + "clusterConsensuses_Labels.txt", header=None).values.ravel()
clusters = clusters[nzPos][top]
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t")
zScoresPerCluster = []
pvals = []
for i in range(clusters.max()+1):
    inClust = clusters == i
    corr = np.corrcoef(countsScaled[:, inClust].T)
    zScoresPerCluster.append(countsScaled[:, inClust])
    corrNull = []
    for j in range(1+int(1e5/inClust.sum()**2)):
        null = np.random.permutation(inClust)
        corrNull.append(np.corrcoef(countsScaled[:, null].T).ravel())
    corrNull = np.concatenate(corrNull)
    pvals.append(ttest_ind(corr.ravel(), corrNull.ravel(), equal_var=False, alternative="greater")[1])
    plt.figure()
    plt.hist(corr.ravel(), 20, alpha=0.5, density=True)
    plt.hist(corrNull.ravel(), 20, alpha=0.5, density=True)
    plt.show()
    sns.clustermap(corr, method="ward", vmin=0, vmax=1)
    # sns.clustermap(corrNull, method="ward", vmin=0, vmax=1)
    # sns.clustermap(countsScaled[:, inClust], method="ward", row_colors=colors)
    if i > 10:
        break
    
import seaborn as sns

sns.clustermap(corr, method="ward", vmin=0, vmax=1)
sns.clustermap(corrNull[:, :10000], method="ward", vmin=0, vmax=1)
sns.clustermap(countsScaled[:, inClust], method="ward", row_colors=colors, vmin=-3, vmax=3)
'''