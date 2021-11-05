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
# %%
np.random.seed(42)
progCase = dict()
statsCase = dict()
studiedConsensusesCase = dict()
    case = "TCGA-LAML"

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
    # Read survival information
    survived = (annotation["vital_status"] == "Alive").values
    timeToEvent = annotation["days_to_last_follow_up"].where(survived, annotation["days_to_death"])
    # Drop rows with missing survival information
    toDrop = timeToEvent.index[timeToEvent == "'--"]
    boolIndexing = np.logical_not(np.isin(timeToEvent.index, toDrop))
    timeToEvent = timeToEvent.drop(toDrop)
    timeToEvent = timeToEvent.astype("float")
    survived = survived[boolIndexing]
    annotation.drop(toDrop)
    dlFiles = dlFiles[boolIndexing]
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
    allCounts = np.concatenate(counts, axis=1)
    # Remove low counts + scran deconvolution normalization
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    scran = importr("scran")


    countsNorm = allCounts.T / np.mean(allCounts, axis=0)[:, None]
    countsNorm = countsNorm / np.min(countsNorm[countsNorm.nonzero()])
    nz = np.sum(countsNorm >= 1, axis=0) > 2
    countsNorm = countsNorm[:, nz]
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

    quantileSize = 50
    subsample = np.random.choice(countsNorm.ravel()[allCounts[nz].T.ravel().nonzero()], 1000000)
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
    # Evaluate significance of variance with a permutation test per quantile of mean expression 
    for i in range(n):
        inBin = (assigned) == i
        if inBin.sum() > 0.5:
            rand = permutationVariance(logCounts[:, inBin])
            pvals[inBin] = computeEmpiricalP(var[inBin], rand)
    top = qvalue(pvals)[1] < 0.05
    c = np.zeros((len(pvals), 3)) + np.array([0.0,0.0,1.0])
    c[top] = [1.0,0.0,0.0]
    plt.figure(dpi=500)
    plt.scatter(mean, var, c=c, s=0.5, alpha=0.5, linewidths=0.0)
    plt.show()
    # %%
    # Yeo-johnson transform and scale to unit variance
    countsScaled = power_transform(countsNorm[:, top])
    # %%
    from rpy2.robjects import r, pandas2ri
    import rpy2.robjects as ro
    from sklearn.preprocessing import scale
    pandas2ri.activate()
    maxstat = importr("maxstat")
    survival = importr("survival")
    df = pd.DataFrame()
    df["Dead"] = np.logical_not(survived[np.isin(timeToEvent.index, order)])
    df["TTE"] = timeToEvent.loc[order].values + 1e-5
    df[np.arange(countsScaled.shape[1])] = countsScaled
    df.index = order
    df = df.copy()
    # %%
    from sklearn.cluster import AgglomerativeClustering, MeanShift
    import kaplanmeier as km
    import lifelines as ll

    stats = []
    cutoffs = []
    notDropped = []
    for i in range(top.sum()):
        cph = ll.CoxPHFitter()
        cph.fit(df[[i, "TTE", "Dead"]], duration_col="TTE", event_col="Dead", robust=True)
        stats.append(cph.summary)

    stats = pd.concat(stats)
    pvals = np.array(stats["p"])
    pvals.min()
    statsCase[case] = stats
    # %%
    from lib.qvalue import qvalue
    qvals = qvalue(pvals)

    # %%
    consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
    topLocs = consensuses[nz][top].iloc[(qvals < 0.05).nonzero()[0]]
    topLocs.to_csv(paths.tempDir + f"topLocs{case}_prog.bed", sep="\t", header=None, index=None)
    studiedConsensuses[case] = nz.nonzero()[0][top]
    progArray = np.zeros(len(consensuses), dtype=bool)
    progCase[case] = nz.nonzero()[0][top][qvals < 0.05]
    progArray[progCase[case]] = True
# %%
