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
from scipy.special import expit
from sklearn.preprocessing import StandardScaler, RobustScaler, power_transform

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
# %%
allCounts = np.concatenate(counts, axis=1)
# bgCounts = np.concatenate(countsBG, axis=1)
# %%
# Keep tumoral samples
kept = np.isin(order, annotation.index)
allCounts = allCounts[:, kept]
# bgCounts = bgCounts[:, kept]
allReads = allReads[kept]
annotation = annotation.loc[np.array(order)[kept]]
kept = np.logical_not(annotation["Sample Type"] == "Solid Tissue Normal")
annotation = annotation[kept]
allCounts = allCounts[:, kept]
# bgCounts = bgCounts[:, kept]
allReads = allReads[kept]
# %%
'''
meanPolII = np.mean(allCounts, axis=0)
meanBG = np.mean(bgCounts, axis=0)
plt.figure(dpi=500)
plt.xlabel("")
plt.hist(meanPolII/meanBG, 20)
print("Median enrichment vs background :", np.median(meanPolII/meanBG))
'''
# %%
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scran = importr("scran")
# %%
# Remove low counts + scran deconvolution normalization
countsNorm = allCounts.T / np.mean(allCounts, axis=0)[:, None]
countsNorm = countsNorm / np.min(countsNorm[countsNorm.nonzero()])
nzPos = np.mean(countsNorm, axis=0) > 1
countsNorm = countsNorm[:, nzPos]
scale = np.array(scran.calculateSumFactors(countsNorm.T))
countsNorm = countsNorm / scale[:, None]
countsNorm = countsNorm / np.min(countsNorm[countsNorm.nonzero()])
# countsNorm = countsNorm / np.min(countsNorm[countsNorm.nonzero()])

dec = scran.modelGeneVar(np.log2(1+countsNorm.T))
mean = np.array(dec.slots["listData"].rx("mean")).ravel()
var = np.array(dec.slots["listData"].rx("total")).ravel()
fdr = np.array(dec.slots["listData"].rx["FDR"]).ravel()
pval = np.array(dec.slots["listData"].rx["p.value"]).ravel()
# %%
top = pval < 0.05
c = np.zeros((len(fdr), 3)) + np.array([0.0,0.0,1.0])
c[top] = [1.0,0.0,0.0]
plt.figure(dpi=500)
plt.scatter(mean, var, c=c, s=0.5, linewidths=0.0)
plt.show()
# %%
# Yeo-johnson transform and scale to unit variance
countsScaled = power_transform(countsNorm[:, top])
# countsScaled = StandardScaler().fit_transform(np.log2(1+countsNorm[:, top]))
# countsScaled = StandardScaler().fit_transform(countsNorm[:, top])
# countsScaled = sTransform(countsNorm[:, top], 0.55)
plt.figure()
plt.hist(countsScaled.ravel(), 20)
plt.yscale("log")
plt.xlabel("Z-scores")
plt.show()
# %%
# Outliers :/
plt.figure(dpi=500)
plt.boxplot(countsScaled[np.random.choice(len(countsScaled), 100, replace=False)].T,showfliers=False)
plt.xlabel("100 samples")
plt.ylabel("Distribution of expression z-scores per sample")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()
# %%
import umap
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch
embedding = umap.UMAP(metric="correlation", low_memory=False).fit_transform(countsScaled)
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
plt.gca().set_aspect(xScale/yScale)
plt.show()
plt.figure(dpi=500)
patches = []
for i in np.unique(cancerType):
    legend = Patch(color=palette[i], label=eq[i])
    patches.append(legend)
plt.legend(handles=patches)
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