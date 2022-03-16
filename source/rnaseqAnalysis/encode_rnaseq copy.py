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
scran = importr("scran")
numpy2ri.activate()
counts = allCounts
sf = scran.calculateSumFactors(counts.T, scaling=allReads[:, None])
countsNorm = counts/np.array(sf)[:, None]
nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=5, expMin=3)
countsNorm = countsNorm[:, nzCounts]
logCounts = np.log(1+countsNorm)
# %%
countsSel = counts[:, nzCounts]
n_i = np.sum(countsSel, axis=1)
n_i = n_i / np.median(n_i)
countsSel = countsSel / n_i[:, None]
# countsSel = countsNorm
# %%
# Estimate 
import statsmodels.discrete.discrete_model as discrete_model
from statsmodels.genmod.families.family import NegativeBinomial, Poisson, Gaussian
from statsmodels.genmod.families import family, links
from statsmodels.genmod.generalized_linear_model import GLM
import warnings
warnings.filterwarnings("error")
geneSubSample = 20000
fittedParams = []
worked = []
np.random.seed(42)
shuffled = np.random.permutation(countsSel.shape[1])[:geneSubSample]
params = np.ones_like(countsSel[:, 0])
for i in shuffled:
    if i % 100 == 0:
        print(i)
    try:
        model = discrete_model.NegativeBinomial(countsSel[:, i], params)
        fit = model.fit(disp=0, method="nm", ftol=1e-9, maxiter=500)
        if np.abs(np.log10(fit.params[1])) > 2:
            continue
        fittedParams.append(fit.params)
        worked.append(i)
    except:
        continue
warnings.filterwarnings("default")
fittedParams = np.array(fittedParams)
b0 = np.array(fittedParams[:, 0])
alphas = np.array(fittedParams[:, 1])
means = np.mean(countsSel[:, worked], axis=0)

# %%

# %%
# Estimate overdispersion in function of mean
import scipy.interpolate as si
m = np.mean(countsSel,axis=0)
bins = np.percentile(m, np.linspace(0,100,25))
bins[0] = 0
bins[-1] += 1000
assigned = np.digitize(m, bins)
alphaReg = []
trueMeans = []
for i in np.unique(assigned):
    subset = countsSel[:, assigned==i]
    subSetMean = np.mean(subset)
    trueMeans.append(subSetMean)
    # Rescale to have same mean
    subset = (subset/np.mean(subset, axis=0) * subSetMean)
    subsetProbeVar = subset.var(axis=0)
    subset = subset[(subset <= np.maximum(np.percentile(subset, 90),5))]    
    subset = subset[np.random.choice(len(subset), min(100000, len(subset)), replace=False)]
    model = discrete_model.NegativeBinomial(subset, np.ones_like(subset))
    fit = model.fit(disp=0, method="nm", maxiter=500)
    alphaReg.append(fit.params[1])
fittedAlpha = si.interp1d(trueMeans, alphaReg, bounds_error=False, fill_value=(alphaReg[0], alphaReg[-1]))
regAlpha = fittedAlpha(np.mean(countsSel, axis=0))
plt.figure(dpi=500)
plt.plot(regAlpha[np.argsort(np.mean(countsSel, axis=0))])
plt.scatter(np.argsort(np.argsort(means))*countsSel.shape[1]/len(alphas), alphas, s=0.5, linewidths=0, c="red")
plt.yscale("log")
plt.xlabel("Pol II ranked mean expression")
plt.ylabel("Alpha (overdispersion)")

# %%
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
n_i = np.sum(countsSel, axis=1)
rps = []
rds = []
errors = []
deviances = []
alphaReg = np.median(alphas)
worked = []
params = np.ones_like(countsSel[:, 0])
for i in range(countsSel.shape[1]):
    if i % 1000 == 0:
        print(i)
    func = family.NegativeBinomial(alpha=regAlpha[i])
    pred = np.repeat(np.mean(countsSel[:, i]), 510)
    rds.append(func.resid_anscombe(countsSel[:, i], pred))
    deviances.append(func.deviance(countsSel[:, i], pred))
    worked.append(i)

rds=np.array(rds)
rds = np.nan_to_num(rds)
deviances = np.nan_to_num(deviances)
# %%
pvals = chi2.sf(deviances, 509)
hv = fdrcorrection(pvals, alpha=0.05)[0]
hv.sum()
# %%
v = np.var(countsSel[:, :len(rds)], axis=0)
m = np.mean(countsSel[:, :len(rds)], axis=0)
c = np.array([[0.0,0.0,1.0]]*(len(rds)))
c[hv] = [1.0,0.0,0.0]
plt.figure(dpi=500)
plt.scatter(m, v, s = 0.2, linewidths=0, c=c)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Pol II probe mean")
plt.ylabel("Pol II probe variance")
# %%
def lrtTestPCA(matrix):
    model = PCA(200, whiten=True)
    decomp = model.fit_transform(matrix)
    mses=[np.sum([Gaussian().loglike(matrix[j, :], 0) for j, r in enumerate(matrix)])]
    for i in range(1, 200):
        recons = np.dot(decomp[:, :i], np.sqrt(model.explained_variance_[:i, None]) * model.components_[:i],)
        mse = np.sum([Gaussian().loglike(matrix[j, :], r) for j, r in enumerate(recons)])
        print(mses[-1]-mse)
        p = chi2.sf(-mses[-1]+mse, np.prod(model.components_[0:1].shape))
        print(p)
        if p > 0.05:
            break
        mses.append(mse)
        print(i, mse)
    return PCA(i-1, whiten=True).fit_transform(matrix)
from sklearn.decomposition import PCA
from lib.jackstraw.permutationPA import permutationPA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
feat = StandardScaler().fit_transform(rds[hv].T)
# bestRank = permutationPA(feat, max_rank=min(100, len(rankQT)))
# model = PCA(42, whiten=True)
decomp = lrtTestPCA(feat)
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5,
                     random_state=42, low_memory=False, metric="correlation").fit_transform(feat)
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
from scipy.cluster import hierarchy
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(feat, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(feat.T, "correlation")
# %%
# Plot dendrograms
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
clippedSQ= np.sqrt(countsSel)
plot_utils.plotHC(clippedSQ.T[hv], annotation.loc[order]["Annotation"], countsSel.T[hv],  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_hvg.pdf")
# %%
colOrderAll, colLinkAll = matrix_utils.threeStagesHClinkage(rds, "correlation")
# Plot dendrograms
plt.figure(dpi=500)
hierarchy.dendrogram(colLinkAll, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_col_dendrogram.pdf")
plt.show()
clippedSQ= np.sqrt(countsSel)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["Annotation"], countsSel.T,  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrderAll)
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
