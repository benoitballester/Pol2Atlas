import numpy as np
from scipy.stats.mstats import gmean


def RLE(counts):
    m = gmean(counts, axis=0)
    c1 = np.where(m > 0.9, counts / m, np.nan)
    scales = np.nanmedian(c1, axis=1)[:, None]
    return counts / scales


def fpkm(counts):
    return counts/np.sum(counts, axis=1)[:, None]


def fpkmUQ(counts):
    countsUQ = np.zeros_like(counts, dtype="float")
    for i in range(len(counts)):
        countsUQ[i] = counts[i] / np.percentile(counts[i][counts[i].nonzero()],75)
    countsUQ /= np.min(countsUQ[countsUQ.nonzero()])
    return countsUQ

def k3N(counts):
    pass