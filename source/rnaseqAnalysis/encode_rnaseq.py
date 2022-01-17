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
from scipy.spatial.distance import dice
import matplotlib as mpl
import fastcluster
import sklearn.metrics as metrics

countDir = "/scratch/pdelangen/projet_these/outputPol2/rnaseq/encode_counts/"
try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/")
except FileExistsError:
    pass
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
ranks = rankdata(counts, "average", axis=1)
# %%
# Feature selection based on rank variability
selected = rnaseqFuncs.variableSelection(ranks, plot=True)
# Apply normal quantile transform per gene
rankQT = rnaseqFuncs.quantileTransform(ranks)
# %%
from sklearn.decomposition import PCA
decomp = matrix_utils.autoRankPCA(rankQT[:, selected], whiten=True)
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5,
                     random_state=42, low_memory=False, metric="euclidean").fit_transform(decomp)
plt.figure(figsize=(10,10), dpi=500)
palette, colors = plot_utils.getPalette(ann)
plot_utils.plotUmap(embedding, colors)
patches = []
for i in np.unique(ann):
    legend = Patch(color=palette[i], label=eq[i])
    patches.append(legend)
plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/umap_samples.pdf")
plt.show()
# %%
clustsPol2 = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt",dtype=int)[nzCounts]
nClusts = np.max(clustsPol2)+1
nAnnots = len(eq)
zScores = np.zeros((nClusts, nAnnots))
for i in range(nAnnots):
    hasAnnot = ann == i
    sd = np.std(np.percentile(ranks[hasAnnot], 95, axis=0))
    expected = np.mean(np.percentile(ranks[hasAnnot], 95, axis=0))
    for j in range(nClusts):
        inClust = clustsPol2 == j
        notInClust = np.logical_not(clustsPol2 == j)
        observed = np.mean(np.percentile(ranks[hasAnnot][:, inClust], 95, axis=0))
        zScores[j, i] = (observed-expected)/sd
# %%
rowOrder, colOrder = matrix_utils.HcOrder(zScores)
zClip = np.clip(zScores,0.0,10.0)
zNorm = zClip / np.percentile(zClip, 95)
colorMat = sns.color_palette("vlag", as_cmap=True)(zNorm[rowOrder][:, colOrder])
plt.figure(dpi=300)
plt.imshow(colorMat)
plt.gca().set_aspect(1)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/signalPerClustPerAnnot.pdf")
plt.show()
plt.figure(figsize=(6, 1), dpi=300)
norm = mpl.colors.Normalize(vmin=0, vmax=np.percentile(zClip, 95))
cb = mpl.colorbar.ColorbarBase(plt.gca(), sns.color_palette("vlag", as_cmap=True), norm, orientation='horizontal')
cb.set_label("95th percentile Z-score")
plt.tight_layout()
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/signalPerClustPerAnnot_colorbar.pdf")
plt.show()
# %%
clusteredEncode = matrix_utils.graphClustering(ranks.T, "correlation", k=50, restarts=10)
ami = metrics.adjusted_mutual_info_score(clusteredEncode, clustsPol2)
# %%
agreementMatrix = np.zeros((np.max(clusteredEncode)+1, np.max(clustsPol2)+1))
for i in range(np.max(clusteredEncode)+1):
    for j in range(np.max(clustsPol2)+1):
        inEncodeClust = clusteredEncode == i
        inPol2Clust = clustsPol2 == j
        agreementMatrix[i,j] = 1-dice(inEncodeClust, inPol2Clust)
colorMat = sns.color_palette("vlag", as_cmap=True)(agreementMatrix/np.max(agreementMatrix))
plt.figure(dpi=300)
plt.imshow(colorMat)
plt.ylabel("ENCODE clusters")
plt.xlabel("Pol II clusters")
plt.title(f"Dice similarity between clusters\nClustering AMI : {np.round(ami*1000)/1000}")
plt.gca().set_aspect(1)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/clusteringAgreement.pdf")
plt.show()
plt.figure(figsize=(6, 1), dpi=300)
norm = mpl.colors.Normalize(vmin=0, vmax=np.max(agreementMatrix))
cb = mpl.colorbar.ColorbarBase(plt.gca(), sns.color_palette("vlag", as_cmap=True), norm, orientation='horizontal')
cb.set_label("Dice similarity")
plt.tight_layout()
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/clusteringAgreement_colorbar.pdf")
plt.show()
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
                     random_state=42, low_memory=False, metric="correlation").fit_transform(rankQT.T)
# %%
from scipy.special import erf
rg01 = (erf(rankQT.T)*0.5+0.5) > 0.95
signalPerCategory = np.zeros((np.max(ann)+1, embedding.shape[0]))
for i in range(np.max(ann)+1):
    signalPerCategory[i, :] = np.mean(rg01[:, ann == i], axis=1)
signalPerCategory /= np.sum(signalPerCategory, axis=0) + 1e-15
maxSignal = np.argmax(signalPerCategory, axis=0)
entropy = np.sum(-signalPerCategory*np.log(signalPerCategory+1e-15), axis=0)
normEnt = entropy / (-np.log(1.0/signalPerCategory.shape[0]+1e-15))
# gini = (1 - np.sum(np.power(1e-7+signalPerCategory/(1e-7+np.sum(signalPerCategory,axis=0)), 2),axis=0))
# Retrieve colors based on point annotation
palette, colors = plot_utils.getPalette(maxSignal)
colors = (1.0 - normEnt[:,None]) * colors + normEnt[:,None] * 0.5
plt.figure(figsize=(10,10), dpi=500)
plot_utils.plotUmap(embedding, colors)
patches = []
for i in np.unique(ann):
    legend = Patch(color=palette[i], label=eq[i])
    patches.append(legend)
plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/umap_consensuses.pdf")
plt.show()
# %%
