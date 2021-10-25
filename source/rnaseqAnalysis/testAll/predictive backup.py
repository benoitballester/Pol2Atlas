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
case = "TCGA-KIRC"

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
allCounts = np.concatenate(counts, axis=1)

# %%
scale = np.mean(allCounts, axis=0)
countsNorm = allCounts.T / scale[:, None]
largest = countsNorm[scale.argmin()]
countsNorm = countsNorm / np.min(countsNorm[countsNorm.nonzero()])
# %%
import scipy.stats as ss
from scipy import interpolate
from statsmodels.stats.multitest import fdrcorrection, multipletests
import numba as nb

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

quantileSize = 50
subsample = np.random.choice(countsNorm.ravel()[allCounts.T.ravel().nonzero()], 100000)
scaler = PowerTransformer()
scaler.fit(subsample.reshape(-1,1))
logCounts = scaler.transform(countsNorm.ravel().reshape(-1,1)).reshape(countsNorm.shape)
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
        rand = permutationVariance(logCounts[:, inBin])
        regVar[i] = np.percentile(var[inBin],50)
        hasDoubleExpected[inBin] = var[inBin] > regVar[i]
        pvals[inBin] = computeEmpiricalP(var[inBin], rand)
top = fdrcorrection(pvals)[1] < 0.05
c = np.zeros((len(pvals), 3)) + np.array([0.0,0.0,1.0])
c[top] = [1.0,0.0,0.0]
plt.figure(dpi=500)
plt.scatter(mean, var, c=c, s=0.5, alpha=0.5, linewidths=0.0)
plt.plot(evals, regVar)
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

embedding = umap.UMAP(min_dist=0.0, metric="correlation", low_memory=False).fit_transform(countsScaled)
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
# Predictive model
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import catboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection



labels = []
for a in annotation["Sample Type"].loc[order]:
    if a == "Solid Tissue Normal":
        labels.append(0)
    else:
        labels.append(1)
labels = np.array(labels)
predictions = np.zeros(len(labels), dtype=int)
for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(countsScaled, labels):
    # Fit power transform on train data only
    x_train = countsScaled[train]
    # Fit classifier on scaled train data
    model = catboost.CatBoostClassifier(iterations=100, rsm=np.sqrt(countsScaled.shape[1])/countsScaled.shape[1],
                                        class_weights=len(labels) / (2 * np.bincount(labels)))
    model.fit(x_train, labels[train])
    # Scale and predict on test data
    x_test = countsScaled[test]
    predictions[test] = model.predict(x_test)
print("Weighted accuracy :", balanced_accuracy_score(labels, predictions))
print("Recall :", recall_score(labels, predictions))
print("Precision :", precision_score(labels, predictions))
df = pd.DataFrame(confusion_matrix(labels, predictions))
df.columns = ["Normal Tissue True", "Tumor True"]
df.index = ["Normal Tissue predicted", "Tumor predicted"]

# %%
# DE Genes
def findDE(counts, labels):
    tumorExpr = counts[labels == 1]
    normalExpr = counts[labels == 0]
    pvals = []
    for i in range(counts.shape[1]):
        pvals.append(mannwhitneyu(normalExpr[:, i], tumorExpr[:, i])[1])
    return fdrcorrection(pvals)[1]

qvals = findDE(countsNorm, labels)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
# consensuses = consensuses[nzPos][top]
topLocs = consensuses.iloc[(qvals < 0.05).nonzero()[0]]
topLocs.to_csv(paths.tempDir + f"topLocs{case}_DE.csv", sep="\t", header=None, index=None)
topLocs
# %%
from lib.utils import overlap_utils
import pyranges as pr

remap_catalog = pr.read_bed(paths.remapFile)
allPolII = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
topDEpr = topLocs
enrichments = overlap_utils.computeEnrichVsBg(remap_catalog, allPolII, topLocs)
enrichments[2].sort_values()[:50]
# %%
