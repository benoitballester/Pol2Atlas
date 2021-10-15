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
from sklearn.preprocessing import power_transform, PowerTransformer
from lifelines.statistics import logrank_test
# %%
case = "TCGA-KIRC"
# %%
# Select only relevant files and annotations
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
annotation = annotation[annotation["project_id"] == case]
annotation = annotation[np.logical_not(annotation["Sample Type"] == "Solid Tissue Normal")]
dlFiles = os.listdir(paths.countDirectory + "500centroid/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
ids = np.array([f.split(".")[0] for f in dlFiles])
inAnnot = np.isin(ids, annotation.index)
ids = ids[inAnnot]
dlFiles = np.array(dlFiles)[inAnnot]
annotation = annotation.loc[ids]
# %%
# Read survival information
survived = (annotation["vital_status"] == "Alive").values
timeToEvent = annotation["days_to_last_follow_up"].where(survived, annotation["days_to_death"])
timeToEvent = timeToEvent.astype("float")
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
# %%
# Remove low counts + scran deconvolution normalization
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scran = importr("scran")


countsNorm = allCounts.T / np.mean(allCounts, axis=0)[:, None]
countsNorm = countsNorm / np.min(countsNorm[countsNorm.nonzero()])
nzPos = np.mean(countsNorm, axis=0) > 1
countsNorm = countsNorm[:, nzPos]
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
logCounts = np.log2(1+countsNorm)
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
        # regVar[i] = np.percentile(var[inBin],5)
        # hasDoubleExpected[inBin] = np.var(logCounts[:, inBin], axis=0) > regVar[i]*2
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
countsScaled = (countsNorm[:, top])
# %%
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
pandas2ri.activate()
maxstat = importr("maxstat")
survival = importr("survival")
df = pd.DataFrame()
df[np.arange(countsScaled.shape[1])] = countsScaled
df["Survived"] = survived[np.isin(timeToEvent.index, order)]
df["TTE"] = timeToEvent.loc[order].values
df.index = order
df = df.copy()
r_dataframe = ro.conversion.py2rpy(df)
# %%
from statsmodels.stats.multitest import fdrcorrection, multipletests
n = 100 # Number of quantiles to compute
minPct = 5
maxPct = 95
evalPcts = np.linspace(minPct, maxPct, n)
pvals = []
cutoffs = []
notDropped = []
ttes = timeToEvent.loc[order].values
events = survived[np.isin(timeToEvent.index, order)]
for i in range(top.sum()):
        i = 335
        e = np.median(countsScaled[:, i][countsScaled[:, i]>0])
        gr1 = countsScaled[:, i] < e
        gr2 = np.logical_not(gr1)
        p = logrank_test(ttes[gr1], ttes[gr2], events[gr1], events[gr2]).p_value
        fml = ro.r(f"Surv(TTE, Survived) ~ X{i}")
        mstat = maxstat.maxstat_test(fml, data=r_dataframe, smethod="LogRank", pmethod="condMC")
        mp = mstat.rx2('p.value')[0]
        break

pvals = np.array(pvals)
# %%
qvals = multipletests(pvals, method="fdr_gbs")[1]
# qvals=pvals
# %%
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensuses = consensuses[nzPos][top].iloc[notDropped]
topLocs = consensuses.iloc[(qvals < 0.05).nonzero()[0]]
topLocs[3] = -np.log10(qvals[[(qvals < 0.05).nonzero()[0]]])
topLocs.sort_values(3, ascending=False, inplace=True)
topLocs.to_csv(paths.tempDir + "topLocsBRCA_prog.csv", sep="\t", header=None, index=None)
topLocs
# %%
import kaplanmeier as km
order = np.argsort(pvals)
for i in order[:5]:
    print(f"Top {i} prognostic")
    plt.figure(dpi=300)
    transfo = np.log2(1+countsNorm[:, top][:, notDropped][:, i])
    plt.hist(transfo,50, density=True)
    plt.vlines(np.log2(1+cutoffs[i]), plt.ylim()[0],plt.ylim()[1], color="red")
    plt.vlines(np.log2(1+np.percentile(countsNorm[:, top][:, notDropped][:, i],90)), 
                plt.ylim()[0],plt.ylim()[1], color="green")
    plt.vlines(np.log2(1+np.percentile(countsNorm[:, top][:, notDropped][:, i],10)), 
                plt.ylim()[0],plt.ylim()[1], color="green")
    plt.xlabel("log2(1 + scran counts)")
    plt.ylabel("Density")
    plt.show()
    groups = countsScaled[:, notDropped][:, i] > cutoffs[i]
    kmf = km.fit(df["TTE"], 
                            df["Survived"], groups)
    km.plot(kmf)
    plt.show()
# %%
