import numpy as np
import pandas as pd
import pyranges as pr
import numba as nb

def countOverlapPerCategory(catalog, query):
    return catalog.intersect(query).as_df()["Name"].value_counts().to_dict()


def countOverlaps(catalog, query):
    return len(catalog.intersect(query))


def computeEnrichVsBg(catalog, query, universe):
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