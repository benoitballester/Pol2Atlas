import numpy as np
from scipy.stats.mstats import gmean
from . import rnaseqFuncs
from scipy.stats import rankdata
from scipy.special import erfinv
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scran = importr("scran")

def deseqNorm(counts):
    m = gmean(counts, axis=0)
    c1 = np.where(m > 1e-15, counts / m, np.nan)
    scales = np.nanmedian(c1, axis=1)[:, None]
    return scales


def fpkm(counts):
    return counts/np.sum(counts, axis=1)[:, None]


def fpkmUQ(counts):
    countsUQ = np.zeros_like(counts, dtype="float")
    sfs = np.zeros(len(countsUQ))
    for i in range(len(counts)):
        sfs[i] = np.percentile(counts[i][counts[i].nonzero()],75)
    return sfs


def lowVarNorm(counts):
    rankedCounts = rankdata(counts, "average", axis=1)
    selected = rnaseqFuncs.variableSelection(rankedCounts, alpha=0.05, plot=True)
    lowVar = np.logical_not(selected)
    extremeExpr = np.mean(counts, axis=0) <= np.percentile(np.mean(counts, axis=0), 95)
    sizeFactors = scran.calculateSumFactors(counts.T[lowVar & extremeExpr])
    return (sizeFactors / np.median(sizeFactors))[:, None], selected