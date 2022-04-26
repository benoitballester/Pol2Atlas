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
import scipy.stats as ss
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection
from scipy.spatial.distance import dice
import matplotlib as mpl
import sklearn.metrics as metrics
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
scran = importr("scran")
deseq = importr("DESeq2")
base = importr("base")
s4 = importr("S4Vectors")
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
counts = allCounts
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=5, expMin=2)
countsNz = allCounts[:, nzCounts]
means = np.mean(countsNz/np.mean(countsNz, axis=1)[:, None], axis=0)
with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
    sf = scran.calculateSumFactors(countsNz.T)
countModel = rnaseqFuncs.RnaSeqModeler().fit(countsNz, sf)
anscombe = countModel.anscombeResiduals
countsNorm = countModel.normed
anscombeResiduals = anscombe
pDev, outliers = countModel.hv_selection()
hv = fdrcorrection(pDev)[0] & outliers
# %%
from statsmodels.genmod.families import family
import scipy.stats as ss
from sklearn.preprocessing import PowerTransformer, StandardScaler
rankMeans = np.argsort(np.argsort(countModel.means))
alphas = countModel.regAlpha
res_devs = countModel.res_dev
fitParams = []
scaledDev = []
for i in range(len(rankMeans)):
    if i%1000 == 0:
        print(i)
    nbMean = countModel.means[i] * countModel.scaled_sf
    n = 1 / alphas[i]
    p = nbMean / (nbMean + alphas[i] * (nbMean**2))
    subset = ss.nbinom.rvs(n,p,size=(3,len(p))).ravel()
    func = family.NegativeBinomial(alpha=alphas[i])
    dev = func.resid_dev(subset, np.tile(nbMean[:, None],3).T.ravel())
    scaler = PowerTransformer().fit(dev.reshape(-1,1))
    scaledDev.append(scaler.transform(res_devs[:, i].reshape(-1,1)).ravel())
    fitParams.append([scaler.lambdas_[0], scaler._scaler.mean_[0], scaler._scaler.scale_[0]])
fitParams = np.array(fitParams)
scaledDev = np.array(scaledDev).T
# %%
dev = np.sum(np.square(scaledDev), axis=0)
pVar = chi2.sf(dev, 509)
hv = pVar < 0.05
# %%
# Compute PCA on the anscombe residuals
# Anscombe residuals have asymptotic convergence to a normal distribution with a null NB distribution
# Pearson and raw residuals are skewed for non-normal models like the NB or Poisson
# The number of components is selected with a permutation test
from sklearn.decomposition import PCA
from lib.jackstraw.permutationPA import permutationPA
from sklearn.preprocessing import StandardScaler
feat = StandardScaler().fit_transform(anscombeResiduals)
bestRank = permutationPA(feat[:, hv], max_rank=min(100, len(anscombeResiduals)))
decomp = PCA(bestRank[0]).fit_transform(feat[:, hv])
matrix_utils.looKnnCV(decomp, ann, "correlation",1)
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5,
                     random_state=42, low_memory=False, metric="correlation").fit_transform(decomp)
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
rowOrder, rowLink = matrix_utils.twoStagesHClinkage(decomp, "euclidean")
colOrder, colLink = matrix_utils.threeStagesHClinkage(scaledDev[:, hv].T, "correlation")
# %%
# Plot dendrograms
from scipy.cluster import hierarchy
plt.figure(dpi=500)
hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_hvg_col_dendrogram.pdf")
plt.show()
plt.close()
# Plot dendrograms
plt.figure(dpi=500)
hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1, orientation="left")
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_row_dendrogram.pdf")
plt.show()
# %%
meanNormed = countsNorm/np.mean(countsNorm, axis=0)
epsilon = 1/np.nanmax(np.log(meanNormed), axis=0)
clippedSQ= np.log(1+countsNorm)
plot_utils.plotHC(clippedSQ.T[hv], annotation.loc[order]["Annotation"], countsNorm.T[hv],  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_hvg.pdf")
# %%
colOrderAll, colLinkAll = matrix_utils.threeStagesHClinkage(anscombeResiduals, "correlation")
# Plot dendrograms
plt.figure(dpi=500)
hierarchy.dendrogram(colLinkAll, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_col_dendrogram.pdf")
plt.show()
clippedSQ= np.log(1+countsNorm)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["Annotation"], countsNorm.T,  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrderAll)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM.pdf")
# %%
# Comparison Pol II biotype vs RNA-seq
import pickle
polIIMerger = pickle.load(open(paths.outputDir + "merger", "rb"))
# %%
annotationDf = pd.read_csv(paths.annotationFile, sep="\t", index_col=0)
annotationsP2, eq2 = pd.factorize(annotationDf.loc[polIIMerger.labels]["Annotation"],
                                sort=True)
signalPerCategory = np.zeros((np.max(annotationsP2)+1, len(polIIMerger.embedding[0])))
signalPerAnnot = np.array([np.sum(polIIMerger.matrix[:, i == annotationsP2]) for i in range(np.max(annotationsP2)+1)])
for i in range(np.max(annotationsP2)+1):
    signalPerCategory[i, :] = np.sum(polIIMerger.matrix[:, annotationsP2 == i], axis=1) / signalPerAnnot[i]
signalPerCategory /= np.sum(signalPerCategory, axis=0)
# %%
rnaseqPerCategory = np.zeros((np.max(ann)+1, len(countsNorm[1])))
for i in range(np.max(ann)+1):
    rnaseqPerCategory[i, :] = np.mean(countsNorm.T[:, ann == i], axis=1)
rnaseqPerCategory /= np.sum(rnaseqPerCategory, axis=0)
signalPerCategory = signalPerCategory[:, nzCounts]
# %%
try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/polII_vs_rnaseq/")
except FileExistsError:
    pass
for test in eq2:
    idx1 = list(eq).index(test)
    idx2 = list(eq2).index(test)
    sig1 = signalPerCategory[idx2] + np.random.normal(size=signalPerCategory[idx2].shape)*0.003
    sig2 = rnaseqPerCategory[idx1]
    plt.figure(dpi=500)
    plt.scatter(sig1, sig2, s=0.1, linewidths=0.0)
    plt.xlabel("Pol II probe % of biotype (+gaussian noise to unstack points)")
    plt.ylabel("Fraction of reads in biotype")
    plt.title(test)
    plt.savefig(paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/{test}.pdf")
    plt.close()
# %%
# Boxplot Top 50% pct Pol II vs bottom 50% Pol II Biotype
for test in eq2:
    idx1 = list(eq).index(test)
    idx2 = list(eq2).index(test)
    majoritaryPol2 = signalPerCategory[idx2] > 0.5
    signal = pd.DataFrame(rnaseqPerCategory[idx1], columns=["Fraction of RNA-seq reads in probe"])
    signal["Category"] = np.where(signalPerCategory[idx2] > 0.5, "> 50% of probe biotype", "< 50% of probe biotype")
    upper = rnaseqPerCategory[idx1][signalPerCategory[idx2] > 0.5]
    lower = rnaseqPerCategory[idx1][signalPerCategory[idx2] <= 0.5]
    stat, p = ss.mannwhitneyu(upper, lower)
    plt.figure(dpi=500)
    sns.boxplot(data=signal, x="Category", y="Fraction of RNA-seq reads in probe", showfliers=False)
    sns.stripplot(data=signal, x="Category", y="Fraction of RNA-seq reads in probe", dodge=True, 
                edgecolor="black", jitter=1/4, alpha=1.0, s=0.5)
    plt.title(test + f" (p-value: {p}, direction: {np.sign(np.median(upper)-np.median(lower))})")
    plt.savefig(paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/boxplot_{test}.pdf")
    plt.close()
# %%
# HM FC enrich Top 50% pct Pol II vs bottom 50% Pol II Biotype
resultMat = []
x = []
for test in eq:
    res_annot_encode = []
    y = []
    for test2 in eq2:
        idx1 = list(eq).index(test)
        idx2 = list(eq2).index(test2)
        upper = rnaseqPerCategory[idx1][signalPerCategory[idx2] > 0.5]
        lower = rnaseqPerCategory[idx1][signalPerCategory[idx2] <= 0.5]
        fc = np.nan_to_num(np.log2(np.mean(upper)/np.mean(lower)))
        res_annot_encode.append(fc)
        y.append(test2)   
    x.append(test)
    resultMat.append(res_annot_encode)
matching = np.isin(x, y)
resultMat = pd.DataFrame(resultMat, x, y)
resultMat = resultMat[np.sort(y)]
resultMat = resultMat.loc[np.sort(np.array(x)[matching])]
plt.figure(dpi=500)
sns.heatmap(resultMat, cmap="vlag", vmin=-1.0, vmax=1.0,)
plt.xlabel("Pol II annotations")
plt.ylabel("ENCODE annotations")
plt.title("log2(Reads on probe > 50% Pol Biotype / Reads on probe < 50% Pol Biotype)")
plt.savefig(paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/heatmap_fc.pdf", bbox_inches="tight")
plt.close()
# %%
rowOrder = np.argsort(ann)
topCat = rnaseqPerCategory.argmax(axis=0)
signalTopCat = -rnaseqPerCategory[(topCat,range(len(topCat)))]
colOrder = np.lexsort((signalTopCat, topCat))
meanNormed = countsNorm/np.mean(countsNorm, axis=0)
epsilon = 1/np.nanmax(np.log(meanNormed), axis=0)
clippedSQ= np.log(1+countsNorm)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["Annotation"], countsNorm.T,  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_autorank.pdf")
plt.close()
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
import scipy.stats as ss
clustsPol2 = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt",dtype=int)[nzCounts]
nClusts = np.max(clustsPol2)+1
nAnnots = len(eq)
zScores = np.zeros((nClusts, nAnnots))
filteredMat = (countsNorm / np.mean(countsNorm, axis=0))
for i in range(nAnnots):
    hasAnnot = ann == i
    subset = np.copy(filteredMat[hasAnnot])
    subset2 = np.copy(filteredMat[np.logical_not(hasAnnot)])
    
    for j in range(nClusts):
        inClust = clustsPol2 == j
        expected = np.mean(subset2[:, inClust], axis=0)+1
        observed = np.mean(subset[:, inClust], axis=0)+1
        zScores[j, i] = np.log2(observed/expected)

# %%
rowOrder, colOrder = matrix_utils.HcOrder(np.nan_to_num(zScores))
rowOrder = np.loadtxt(paths.outputDir + "clusterBarplotOrder.txt").astype(int)
zClip = np.clip(zScores,0.0,10.0)
zNorm = np.clip(zClip,0.0,1.0)
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
