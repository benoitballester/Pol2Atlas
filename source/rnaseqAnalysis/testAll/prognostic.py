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
from lib.qvalue import qvalue
import scipy.stats as ss
from scipy import interpolate
from statsmodels.stats.multitest import fdrcorrection, multipletests
import numba as nb
import kaplanmeier as km
import lifelines as ll

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
# %%
np.random.seed(42)
allAnnots = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
progCase = dict()
statsCase = dict()
studiedConsensusesCase = dict()
# %%
for case in allAnnots["project_id"].unique():
    print(case)
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
    if len(counts) < 20:
        print(case, "not enough samples ({allCounts.shape})")
        continue
    allReads = np.array(allReads)
    allCounts = np.concatenate(counts, axis=1).T
    # Remove zeros counts and deseq normalization
    nz = np.sum(allCounts >= 0.9, axis=0) > len(allCounts)*0.05
    scale = normRNAseq.deseqNorm(allCounts[:, nz])
    countsNorm = allCounts[:, nz] / scale

    quantileSize = 100
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
    # Yeo-johnson transform and scale to unit variance
    countsScaled = power_transform(countsNorm[:, top])
    df = pd.DataFrame()
    df["Dead"] = np.logical_not(survived[np.isin(timeToEvent.index, order)])
    df["TTE"] = timeToEvent.loc[order].values + 1e-5
    df[np.arange(countsScaled.shape[1])] = countsScaled
    df.index = order
    df = df.copy()

    stats = []
    cutoffs = []
    notDropped = []
    print(f"Evaluating {top.sum()} peaks")
    # Compute univariate cox proportionnal hazards p value for consensuses with high variance
    # and detectable reads in > 5 % experiments
    for i in range(top.sum()):
        try:
            cph = ll.CoxPHFitter(penalizer=1e-6)
            cph.fit(df[[i, "TTE", "Dead"]], duration_col="TTE", event_col="Dead", robust=True)
            stats.append(cph.summary)
        except ll.exceptions.ConvergenceError:
            # If the regression failed to converge assume HR=1.0 and p = 1.0
            dummyDF = pd.DataFrame(data=[[0.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0]], 
                                   columns=['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p', '-log2(p)'], 
                                   index=[i])
            dummyDF.index.name = "covariate"
            stats.append(dummyDF)

    stats = pd.concat(stats)
    pvals = np.array(stats["p"])
    pvals.min()
    statsCase[case] = stats
    qvals = fdrcorrection(pvals)[1]
    consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
    topLocs = consensuses[nz][top].iloc[(qvals < 0.05).nonzero()[0]]
    topLocs.to_csv(paths.tempDir + f"topLocs{case}_prog.bed", sep="\t", header=None, index=None)
    studiedConsensusesCase[case] = nz.nonzero()[0][top]
    progCase[case] = nz.nonzero()[0][top][qvals < 0.05]
# %%
# Retrieve which consensuses are prognostic
hasProg = pd.DataFrame()
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
for case in progCase:
    arr = np.zeros(len(consensuses), dtype=bool)
    arr[progCase[case]] = True
    hasProg[case] = arr
# Establish a threshold for a consensus peak to be globally prognostic.
# Perform random permutation of the choosen consensus and select n such as fpr < 0.05
countsRnd = np.zeros(hasProg.shape[1])
for i in range(100):
    shuffledDF = np.apply_along_axis(np.random.permutation, 0, hasProg)
    counts = np.bincount(np.sum(shuffledDF, axis=1))/100
    countsRnd[:counts.shape[0]] += counts
countsObs = np.bincount(np.sum(hasProg, axis=1))
for threshold in range(len(countsObs)):
    randomSum = np.sum(countsRnd[threshold:])
    fpr = randomSum / (np.sum(countsObs[threshold:])+randomSum)
    print(threshold, fpr)
    if fpr < 0.05:
        break
globallyProg = consensuses[hasProg.sum(axis=1) >= threshold]
globallyProg.to_csv(paths.tempDir + f"globallyProg.bed", sep="\t", header=None, index=None)
# %%
from lib.pyGREAT import pyGREAT
enricher = pyGREAT(oboFile=paths.GOfolder + "/go_eq.obo", geneFile=paths.gencode, 
                   geneGoFile=paths.GOfolder + "/goa_human.gaf")
import pyranges as pr
from lib.utils import overlap_utils
remap_catalog = pr.read_bed(paths.remapFile)
# %%
from lib.utils import matrix_utils 
goClass = "molecular_function"
goEnrich = enricher.findEnriched(globallyProg, consensuses)
print(goEnrich[goClass][2][(goEnrich[goClass][2] < 0.05) & (goEnrich[goClass][1] > 2)].sort_values()[:25])
hasEnrich = goEnrich[goClass][2][(goEnrich[goClass][2] < 0.05) & (goEnrich[goClass][3] >= 2)]
clustered = matrix_utils.graphClustering(enricher.matrices[goClass].loc[hasEnrich.index], "dice", 
                                        disconnection_distance=1.0, r=1.0, k=3, restarts=10)
topK = 5
maxP = []
for c in np.arange(clustered.max()+1):
    inClust = hasEnrich.index[clustered == c]
    maxP.append(hasEnrich[inClust].min())
orderedP = np.argsort(maxP)
for c in orderedP:
    inClust = hasEnrich.index[clustered == c]
    strongest = hasEnrich[inClust].sort_values()[:topK]
    if len(strongest) > 0:
        print("-"*20)
        print(strongest)
# %%
enrichments = overlap_utils.computeEnrichVsBg(remap_catalog, consensuses, globallyProg)
orderedP = np.argsort(enrichments[1])[::-1]
enrichments[1][orderedP][enrichments[2][orderedP] < 0.05][:25]
# %%
