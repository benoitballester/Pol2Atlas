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
from sklearn.preprocessing import power_transform, PowerTransformer, QuantileTransformer
# %%
case = "TCGA-PRAD"

# Select only relevant files and annotations
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
annotation = annotation[annotation["project_id"] == case]
dlFiles = os.listdir(paths.countDirectory + "500centroid/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
ids = np.array([f.split(".")[0] for f in dlFiles])
inAnnot = np.isin(ids, annotation.index)
ids = ids[inAnnot]
dlFiles = np.array(dlFiles)[inAnnot]
annotation = annotation.loc[ids]

# %%
# Read files and setup data matrix
counts = []
allReads = []
order = []
for f in dlFiles:
    try:
        fid = f.split(".")[0]
        status = pd.read_csv(paths.countDirectory + "500centroid/" + fid + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(paths.countDirectory + "500centroid/" + f, header=None, skiprows=2).values)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(fid)
    except:
        continue
allReads = np.array(allReads)
# %%
allCounts = np.concatenate(counts, axis=1).T
# %%
'''import scipy.stats as ss
from scipy import interpolate
from statsmodels.stats.multitest import fdrcorrection, multipletests
import numba as nb
from lib.qvalue import qvalue
quantileSize = 2000
subsample = np.random.choice(allCounts.ravel()[allCounts.ravel().nonzero()], 100000)
scaler = PowerTransformer()
scaler.fit(subsample.reshape(-1,1))
logCounts = scaler.transform(allCounts.ravel().reshape(-1,1)).reshape(allCounts.shape)
mean = np.mean(logCounts, axis=0)
var = np.var(logCounts, axis=0)
n = int(logCounts.shape[1]/quantileSize)
evals = np.percentile(mean, np.linspace(0,100,n))
evals[-1] += 1e-5 # Slightly extend last bin to include last gene
assigned = np.digitize(mean, evals)
pvals = np.zeros(len(mean))
hasDoubleExpected = np.zeros(len(mean), dtype=bool)
regVar = np.zeros(len(evals))
# Evaluate significance of variance with a permutation test per quantile of mean expression 
for i in range(n):
    inBin = (assigned) == i
    if inBin.sum() > 0.5:
        regVar[i] = np.percentile(var[inBin], 5)
f = interpolate.interp1d(evals, regVar)
regVarPerGene = f(mean)
# p = ss.norm(mean, regVarPerGene).sf(logCounts)

c = np.zeros((len(pvals), 3)) + np.array([0.0,0.0,1.0])
plt.figure(dpi=500)
plt.scatter(mean, var, c=c, s=0.5, alpha=0.5, linewidths=0.0)
plt.scatter(mean, regVarPerGene, s=2, alpha=0.5, linewidths=0.0)
# plt.plot(evals, f(evals))
plt.show()
nonDE = var < regVarPerGene
nonDE = (np.percentile(allCounts, 5, axis=0) > 0.5) & nonDE'''
# %%
scale = normRNAseq.deseqNorm(allCounts)
countsNorm = allCounts / scale
# %%
import scipy.stats as ss
from scipy import interpolate
from statsmodels.stats.multitest import fdrcorrection, multipletests
import numba as nb
from multipy.fdr import qvalue

@nb.njit(parallel=True)
def permutationVariance(df, ntot=50000):
    n = int(ntot/df.shape[1])
    var = np.zeros(n*df.shape[1])
    for i in nb.prange(n):
        shuffled = np.random.permutation(df.ravel()).reshape(df.shape)
        for j in range(df.shape[1]):
            var[i+j*n] = np.var(shuffled[:, j])
    return var

@nb.njit(parallel=True)
def computeEmpiricalP(x, dist):
    pvals = np.zeros(len(x))
    for i in range(len(x)):
        pvals[i] = np.mean(x[i] < dist)
    return pvals

quantileSize = 100
subsample = np.random.choice(countsNorm.ravel()[allCounts.T.ravel().nonzero()], 1000000)
scaler = PowerTransformer()
scaler.fit(subsample.reshape(-1,1))
logCounts = np.log10(1+countsNorm)
mean = np.mean(logCounts, axis=0)
var = np.var(logCounts, axis=0)
n = int(logCounts.shape[1]/quantileSize)
evals = np.percentile(mean, np.linspace(0,100,n))
evals[-1] += 1e-5 # Slightly extend last bin to include last gene
assigned = np.digitize(mean, evals)
pvals = np.ones(len(mean))
regVar = np.zeros(len(evals))
# Evaluate significance of variance with a permutation test per quantile of mean expression 
for i in range(n):
    inBin = (assigned) == i
    if inBin.sum() > 0.5:
        normed = logCounts[:, inBin]/np.mean(logCounts[:, inBin], axis=0)
        normVar = np.var(normed, axis=0)
        nonDE = normVar <= np.percentile(normVar, 95)
        try:
            rand = permutationVariance(normed[:, nonDE])
            pvals[inBin] = computeEmpiricalP(normVar, rand)
            regVar[i] = np.percentile(normVar, 5)
        except: 
            continue
top = fdrcorrection(pvals)[1] < 0.05
c = np.zeros((len(pvals), 3)) + np.array([0.0,0.0,1.0])
c[top] = [1.0,0.0,0.0]
plt.figure(dpi=500)
plt.scatter(mean, var, c=c, s=0.5, alpha=0.5, linewidths=0.0)
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
# Check for outliers
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
from lib.utils import matrix_utils
embedding = umap.UMAP(min_dist=0.0, metric="euclidean", low_memory=True).fit_transform(countsScaled)
# %%
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch
import seaborn as sns

# cancerType, eq = pd.factorize(tcgaProjects["Origin"].loc[project_id])
# cancerType, eq = pd.factorize(annotation["primary_diagnosis"].loc[order])
cancerType, eq = pd.factorize(annotation["Sample Type"].loc[order])
plt.figure(dpi=500)
palette, colors = getPalette(cancerType)
# allReadsScaled = (allReads - allReads.min()) / (allReads.max()-allReads.min())
# colors = sns.color_palette("rocket_r", as_cmap=True)(allReadsScaled)
plt.scatter(embedding[:,0], embedding[:,1], s=min(10.0,100/np.sqrt(len(embedding))),
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
