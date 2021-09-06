import numpy as np
import pandas as pd
import pyranges as pr


def countOverlapPerCategory(catalog, query):
    return catalog.intersect(query).as_df()["Name"].value_counts().to_dict()


def countOverlaps(catalog, query):
    return len(catalog.intersect(query))


def computeEnrichVsBg(catalog, query, universe):
    pass
