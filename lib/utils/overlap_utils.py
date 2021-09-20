import numpy as np
import pandas as pd
import pyranges as pr
import numba as nb

def countOverlapPerCategory(catalog, query):
    return catalog.intersect(query).as_df()["Name"].value_counts().to_dict()


def countOverlaps(catalog, query):
    return len(catalog.intersect(query))


def computeEnrichVsBg(catalog, query, universe):
    """
    Computes an hypergeometric enrichment test on the number of intersections
    for different classes of genomic elements.

    Parameters
    ----------
    catalog: PyRanges
        Elements to find enrichment on.
        PyRanges having the category of the genomic element in the "name" column.
    query: PyRanges
        The genomic regions on interest. Must be contained in universe.
    universe: PyRanges
        The background regions wose intersection count serves as a null
        distribution.
    """
    pass

def computeEnrichForLabels(catalog, universe, labels):
    """
    Computes an hypergeometric enrichment test on the number of intersections
    for different classes of genomic elements and for each group in the universe

    Parameters
    ----------
    catalog: PyRanges
        Elements to find enrichment on.
        PyRanges having the category of the genomic element in the "name" column.
    universe: PyRanges
        Genomic regions.
    labels: array like of integers
        The group each genomic region in universe belongs to.
    """
    pass

@nb.njit(parallel=True)
def addGenomicContext(starts, ends, annots, array):
    for i in nb.prange(len(starts)):
        array[starts[i]:ends[i]] = np.maximum(array[starts[i]:ends[i]], annots[i])


@nb.njit(parallel=True)
def findPrioritaryContext(starts, ends, array):
    contextes = np.zeros(len(starts), array.dtype)
    for i in nb.prange(len(starts)):
        contextes[i] = np.maximum(array[starts[i]:ends[i]])
    return contextes