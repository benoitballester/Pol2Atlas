import numpy as np
import pandas as pd
import pyranges as pr
from scipy.stats import hypergeom
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
from . import plot_utils
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
rs = importr("stats")

def dfToPrWorkaround(df, useSummit=True):
    # There is an issue when converting a dataframe to a pyrange
    # Convert to separate np array beforehand
    if type(df) == pr.pyranges.PyRanges:
        return df
    if useSummit:
        return pr.PyRanges(chromosomes=df.iloc[:,0].values.ravel(), 
                            starts=df.iloc[:,6].values.ravel(), 
                            ends=df.iloc[:,7].values.ravel())
    else:
        return pr.PyRanges(chromosomes=df.loc[:, "Chromosome"].values.ravel(), 
                            starts=df.loc[:, "Start"].values.ravel(), 
                            ends=df.loc[:, "End"].values.ravel())

def getIntersected(query, catalog):
    pass

def countOverlapPerCategory(catalog, query):
    return catalog.overlap(query, how=None).as_df()["Name"].value_counts()


def countOverlaps(catalog, query):
    return len(catalog.overlap(query))

def getHits(query, catalog):
    query.Name = np.arange(len(query))
    return query.intersect(catalog)

def computeEnrichVsBg(catalog, universe, query, useSummit=True):
    """
    Computes an hypergeometric enrichment test on the number of intersections
    for different classes of genomic elements (catalog).

    Parameters
    ----------
    catalog: PyRanges
        Elements to find enrichment on.
        PyRanges having the category of the genomic element in the "name" column.
    query: PyRanges
        The genomic regions on interest. Must be contained in universe for correct
        results.
    universe: PyRanges
        The background regions whose intersection count serves as the expected results.
    
    Returns
    -------
    pvalues: pandas Series
        pvalues for each class of genomic element
    fc: pandas Series
        Fold change for each lass of genomic element
    qvalues: pandas Series
        Benjamini-Hochberg qvalues for each class of genomic element
    k: pandas series
        Number of intersections for each class of genomic element
    """
    # Compute intersections for universe
    refCounts = countOverlapPerCategory(catalog, dfToPrWorkaround(universe))
    allCats = np.array(list(refCounts.keys()))
    pvals = np.zeros(len(allCats))
    fc = np.zeros(len(allCats))
    M = len(universe)
    # Then for the query
    obsCounts = countOverlapPerCategory(catalog, dfToPrWorkaround(query))
    N = len(query)
    # Find hypergeometric enrichment
    k = pd.Series(np.zeros(len(allCats), dtype="int"), allCats)
    isFound = np.isin(allCats, obsCounts.index, assume_unique=True)
    k[allCats[isFound]] = obsCounts
    n = pd.Series(np.zeros(len(allCats), dtype="int"), allCats)
    n[allCats] = refCounts
    # Scipy hyper 
    pvals = np.array(rs.phyper(k.values-1,n.values,M-n.values,N, lower_tail=False))
    pvals = np.nan_to_num(pvals, nan=1.0)
    fc = (k/np.maximum(N, 1e-7))/np.maximum(n/np.maximum(M, 1e-7), 1e-7)
    qvals = fdrcorrection(pvals)[1]
    pvals = pd.Series(pvals)
    pvals.index = allCats
    qvals = pd.Series(qvals)
    qvals.index = allCats
    fc = pd.Series(fc)
    fc.index = allCats
    return pvals, fc, qvals, k, n


def computeEnrichForLabels(catalog, universe, labels, savePathPrefix=None, useSummit=True, fileFmt="pdf"):
    enrichmentsP = []
    enrichmentsFC = []
    for i in np.unique(labels):
        enrichClust = computeEnrichVsBg(catalog, universe, 
                                        universe[labels == i], useSummit)
        enrichmentsP.append(enrichClust[0])
        enrichmentsFC.append(enrichClust[1])
    enrichmentsP = pd.concat(enrichmentsP, axis=1)
    enrichmentsP.columns = np.unique(labels)
    enrichmentsQ = enrichmentsP.copy()
    enrichmentsQ.iloc[:] = fdrcorrection(enrichmentsQ.values.ravel())[1].reshape(enrichmentsQ.shape)
    enrichmentsFC = pd.concat(enrichmentsFC, axis=1)
    enrichmentsFC.columns = np.unique(labels)
    if savePathPrefix is not None:
        enrichmentsFC.to_csv(f"{savePathPrefix}_fc.tsv", sep="\t")
        enrichmentsQ.to_csv(f"{savePathPrefix}_Q.tsv", sep="\t")
    for i in np.unique(labels):
        fig, ax = plt.subplots(figsize=(2,2), dpi=500)
        plot_utils.enrichBarplot(ax, enrichmentsFC[i], enrichmentsQ[i], fcMin=2.0, order_by="qval")
        if savePathPrefix is not None:
            fig.savefig(f"{savePathPrefix}_fc_{i}.{fileFmt}", bbox_inches="tight")
        fig.show()
    return enrichmentsFC, enrichmentsQ, enrichmentsP



    
