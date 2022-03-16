# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib import rnaseqFuncs, normRNAseq
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
# Plot FPKM expr per annotation
palette = pd.read_csv(paths.polIIannotationPalette, sep=",")
palette = dict([(d["Annotation"], (d["r"],d["g"],d["b"])) for r,d in palette.iterrows()])
fpkmExpr = np.sum(allCounts/allReads[:, None], axis=1)*100
df = pd.DataFrame(data=np.concatenate([fpkmExpr[:, None], annotation.loc[order]["Annotation"].ravel()[:, None]], axis=1), columns=["Percentage of mapped reads", "Annotation"])
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, showfliers=False)
sns.stripplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, dodge=True, 
                edgecolor="black", jitter=1/3, alpha=1.0, s=2, linewidth=0.1)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/pctmapped_per_annot.pdf", bbox_inches="tight")
# %%
# Remove undected Pol II probes
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scran = importr("scran")

counts = allCounts
sf = scran.calculateSumFactors(counts.T, scaling=allReads[:, None])
countsNorm = counts/np.array(sf)[:, None]
nzCounts = rnaseqFuncs.filterDetectableGenes(countsNorm, readMin=1, expMin=3)
countsNorm = countsNorm[:, nzCounts]
logCounts = np.log(1+countsNorm)
# %%
# Feature selection based on rank variability
selected = rnaseqFuncs.variableSelection(rankdata(countsNorm, axis=1), plot=False)
# Apply normal quantile transform per gene
rankQT = logCounts - np.mean(logCounts, axis=0)
rankQT = rankQT / np.std(rankQT, axis=0)
# %%
countsSel = counts[:, nzCounts]
# %%
from sklearn.decomposition import PCA
from lib.jackstraw.permutationPA import permutationPA
bestRank = permutationPA(rps[sig].T, max_rank=min(100, len(rankQT)))
decomp = PCA(bestRank[0], whiten=True).fit_transform(rps[sig].T)
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5,
                     random_state=42, low_memory=False, metric="euclidean").fit_transform(decomp)
plt.figure(figsize=(10,10), dpi=500)
annot, palette, colors = plot_utils.applyPalette(annotation.loc[order]["Annotation"],
                                                np.unique(annotation.loc[order]["Annotation"]),
                                                 paths.polIIannotationPalette, ret_labels=True)
plot_utils.plotUmap(embedding, colors)
patches = []
for i, a in enumerate(annot):
    legend = Patch(color=palette[i], label=a)
    patches.append(legend)
plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/umap_samples.pdf")
plt.show()
# %% 
rowOrder = matrix_utils.threeStagesHC(decomp, "correlation")
colOrder = matrix_utils.threeStagesHC(rankQT.T, "correlation")
# %%
plot_utils.plotHC(logCounts.T, annotation.loc[order]["Annotation"], 
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM.pdf")
# %%
annotOHE = pd.get_dummies(ann)
# %%
'''
import statsmodels.discrete.discrete_model as discrete_model
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedGeneralizedPoisson
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import NegativeBinomial
sig = np.percentile(countsNorm, 99, axis=0) > 10
matSig = countsNorm[:, sig]
clustsPol2 = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt",dtype=int)[nzCounts][sig]
nClusts = np.max(clustsPol2)+1
nAnnots = len(eq)
np.random.seed(42)
zScores = np.zeros((nClusts, nAnnots))
for i in range(nClusts):
    print(i)
    inClust = clustsPol2 == i
    countsClust = matSig[:, inClust].T
    subSample = np.random.choice(len(countsClust), min(len(countsClust), 100), replace=False)
    countsClust = countsClust[subSample]
    meanExpr = np.repeat(countsClust.mean(axis=1), len(annotOHE))
    flattenedCounts = countsClust.ravel()
    annotOHETiled = np.tile(annotOHE.T, len(countsClust)).T
    model = discrete_model.NegativeBinomial(flattenedCounts, annotOHETiled, exposure=meanExpr)
    res = model.fit_regularized(disp=0, method="l1", alpha=0.0, trim_mode="off")
    zScores[i] = res.params[:-1]
'''
# %%

clustsPol2 = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt",dtype=int)[nzCounts]
nClusts = np.max(clustsPol2)+1
nAnnots = len(eq)
zScores = np.zeros((nClusts, nAnnots))
sig = np.percentile(countsNorm, 99, axis=0) > 10
filteredMat = (countsNorm > 1.0)[:, sig]
for i in range(nAnnots):
    hasAnnot = ann == i
    sd = np.std(np.percentile(filteredMat[hasAnnot], 95, axis=0))
    expected = np.mean(np.percentile(filteredMat[hasAnnot], 95, axis=0))
    for j in range(nClusts):
        inClust = clustsPol2[sig] == j
        notInClust = np.logical_not(clustsPol2 == j)
        observed = np.mean(np.percentile(filteredMat[hasAnnot][:, inClust], 95, axis=0))
        zScores[j, i] = (observed-expected)/sd

# %%
rowOrder, colOrder = matrix_utils.HcOrder(zScores)
rowOrder = np.loadtxt(paths.outputDir + "clusterBarplotOrder.txt").astype(int)
zClip = np.clip(zScores,0.0,10.0)
zNorm = np.clip(zClip / np.percentile(zClip, 95),0.0,1.0)
plt.figure(dpi=300)
sns.heatmap(zNorm[rowOrder].T[colOrder], cmap="vlag", linewidths=0.1, linecolor='black', cbar=False)
plt.gca().set_aspect(2.0)
plt.yticks(np.arange(len(eq))+0.5, eq[colOrder])
plt.xticks(np.arange(len(rowOrder))+0.5, np.arange(len(rowOrder))[rowOrder], rotation=90, fontsize=6)
plt.xlabel(f"{len(zNorm)} Pol II clusters")
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/signalPerClustPerAnnot.pdf", bbox_inches="tight")
plt.show()
plt.figure(figsize=(6, 1), dpi=300)
norm = mpl.colors.Normalize(vmin=0, vmax=np.percentile(zClip, 95))
cb = mpl.colorbar.ColorbarBase(plt.gca(), sns.color_palette("vlag", as_cmap=True), norm, orientation='horizontal')
cb.set_label("95th percentile Z-score")
plt.tight_layout()
# plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/signalPerClustPerAnnot_colorbar.pdf")
plt.show()
# %%
clusteredEncode = matrix_utils.graphClustering(rankQT.T, "correlation", k=50, restarts=10)
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
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/clusteringAgreement.pdf", bbox_inches="tight")
plt.show()
plt.figure(figsize=(6, 1), dpi=300)
norm = mpl.colors.Normalize(vmin=0, vmax=np.max(agreementMatrix))
cb = mpl.colorbar.ColorbarBase(plt.gca(), sns.color_palette("vlag", as_cmap=True), norm, orientation='horizontal')
cb.set_label("Dice similarity")
plt.tight_layout()
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/clusteringAgreement_colorbar.pdf", bbox_inches="tight")
plt.show()
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
                     random_state=42, low_memory=False, metric="correlation").fit_transform(rankQT.T)
# %%
from scipy.special import erf
mat = countsNorm > np.percentile(countsNorm, 95, axis=0)
signalPerCategory = np.zeros((np.max(ann)+1, embedding.shape[0]))
for i in range(np.max(ann)+1):
    signalPerCategory[i, :] = np.mean(logCounts.T[:, ann == i], axis=1)
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
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/umap_consensuses.pdf", bbox_inches="tight")
plt.show()
