import numpy as np
import pandas as pd
import pyranges as pr
from scipy.stats import hypergeom
from statsmodels.stats.multitest import fdrcorrection
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
rs = importr("stats")

def dfToPrWorkaround(df, useSummit=True):
    # There is an issue when converting a dataframe to a pyrange
    # Convert to separate np array beforehand
    if useSummit:
        return pr.PyRanges(chromosomes=df.iloc[:,0].values.ravel(), 
                            starts=df.iloc[:,6].values.ravel(), 
                            ends=df.iloc[:,7].values.ravel())
    else:
        return pr.PyRanges(chromosomes=df.iloc[:,0].values.ravel(), 
                            starts=df.iloc[:,1].values.ravel(), 
                            ends=df.iloc[:,2].values.ravel())

def getIntersected(query, catalog):
    pass

def countOverlapPerCategory(catalog, query):
    return catalog.overlap(query, how=None).as_df()["Name"].value_counts()


def countOverlaps(catalog, query):
    return len(catalog.overlap(query))

def getHits(query, catalog):
    query.Name = np.arange(len(query))
    return query.intersect(catalog)

def computeEnrichVsBg(catalog, universe, query):
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
    return pvals, fc, qvals, k


def computeEnrichNoBg(catalog, universe, genome):
    """
    Computes an hypergeometric enrichment test on the number of intersections
    for different classes of genomic elements (catalog).

    Parameters
    ----------
    catalog: PyRanges
        Elements to find enrichment on.
        PyRanges having the category of the genomic element in the "name" column.
    query: PyRanges
        The genomic regions of interest. 

    Returns
    -------
    pvalues: pandas Series
        pvalues for each class of genomic element
    fc: pandas Series
        Fold change for each lass of genomic element
    qvalues: pandas Series
        Benjamini-Hochberg qvalues for each class of genomic element
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
    return pvals, fc, qvals

    
