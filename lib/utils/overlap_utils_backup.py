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
    return catalog.intersect(query).as_df()["Name"].value_counts().to_dict()


def countOverlaps(catalog, query):
    return len(catalog.intersect(query))


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
    refCounts = countOverlapPerCategory(catalog, universe)
    allCats = np.array(list(refCounts.keys()))
    pvals = np.zeros(len(allCats))
    fc = np.zeros(len(allCats))
    M = len(universe)
    i = 0
    obsCounts = countOverlapPerCategory(catalog, query)
    N = len(query)
    j = 0
    for c in allCats:
        n = refCounts[c]
        if c in obsCounts.keys():
            k = obsCounts[c]
        else:
            k = 0
        pvals[j] = np.array(rs.phyper(k-1,n,M-n,N, lower_tail=False))[0]
        fc[j] = (k/max(N, 1e-7))/max(n/max(M, 1e-7), 1e-7)
        j += 1
    qvals = fdrcorrection(pvals)[1]
    pvals = pd.Series(pvals)
    pvals.index = allCats
    qvals = pd.Series(qvals)
    qvals.index = allCats
    fc = pd.Series(fc)
    fc.index = allCats
    return pvals, fc, qvals


def computeEnrichForLabels(catalog, universe, labels):
    """
    Computes an hypergeometric enrichment test on the number of intersections
    for different classes of genomic elements in the catalog (name column)
    and for each class (labels) of the universe.

    Parameters
    ----------
    catalog: PyRanges
        Elements to find enrichment on.
        PyRanges having the category of the genomic element in the "name" column.
    universe: PyRanges
        All genomic regions.
    labels: array like
        The group each genomic region in universe belongs to. Can be integer or string.
        It will iterate over each unique label.
    
    Returns
    -------
    pvalues: pandas DataFrame
        pvalues for each label and class of genomic element
    fc: pandas DataFrame
        Fold change for each label and class of genomic element
    qvalues: pandas DataFrame
        Benjamini-Hochberg qvalues for each label and class of genomic element
        
    """
    allLabels = np.unique(labels)
    refCounts = countOverlapPerCategory(catalog, universe)
    allCats = np.array(list(refCounts.keys()))
    pvals = np.zeros((len(allLabels), len(allCats)))
    fc = np.zeros((len(allLabels), len(allCats)))
    M = len(universe)
    i = 0
    for l in allLabels:
        query = universe[labels == l]
        obsCounts = countOverlapPerCategory(catalog, query)
        N = len(query)
        j = 0
        for c in allCats:
            n = refCounts[c]
            if c in obsCounts.keys():
                k = obsCounts[c]
            else:
                k = 0
            pvals[i, j] = hypergeom(M, n, N).sf(k-1)
            fc[i, j] = (k/max(N, 1e-7))/max(n/max(M, 1e-7), 1e-7)
            j += 1
        i += 1
    qvals = fdrcorrection(pvals.ravel())[1].reshape(pvals.shape)
    pvals = pd.DataFrame(pvals)
    pvals.columns = allCats
    pvals.index = allLabels
    qvals = pd.DataFrame(qvals)
    qvals.columns = allCats
    qvals.index = allLabels
    fc = pd.DataFrame(fc)
    fc.columns = allCats
    fc.index = allLabels
    return pvals, fc, qvals
    

    
