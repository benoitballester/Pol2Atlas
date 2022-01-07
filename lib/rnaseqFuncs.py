import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib.utils import plot_utils, matrix_utils
from matplotlib.patches import Patch
from scipy.stats import rankdata, chi2
from scipy.stats import chi2
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection


def filterDetectableGenes(counts, readMin, expMin):
    return np.sum(counts >= readMin, axis=0) >= expMin


def variableSelection(matrix, alpha=0.05, plot=False):
    # Estimate mean and variance for each feature
    m = np.mean(matrix, axis=0)
    v = np.var(matrix, axis=0)
    # Compute a regularized estimate of the variance according to the mean
    fittedSD = np.poly1d(np.polyfit(m, v, 3))(m)
    # Cap variance estimate to min variance estimate
    fittedSD = np.maximum(fittedSD, v.min())
    # Compute normal distribution scaled deviance 
    stat = np.square((matrix - m))/fittedSD
    deviance = np.sum(stat, axis=0)
    # Deviance follows a chi square distribution under null hypothesis
    pvals = chi2.sf(deviance, len(matrix))
    # Select significantly poorly fitted features 
    selected = fdrcorrection(pvals)[1] < alpha
    if plot:
        pct = rankdata(deviance)/len(deviance)
        color = sns.color_palette("magma", as_cmap=True)(pct)
        plt.figure(dpi=500)
        plt.scatter(m, v, c=color, s=0.2, linewidths=0.0)
        plt.gca().set_facecolor((0.5,0.5,0.5))
        plt.scatter(m, fittedSD, s=0.2, linewidths=0.0)
        plt.show()
        plt.figure(dpi=500)
        color = np.array([[0,0,1]]*len(m))
        color[selected] = [1,0,0]
        plt.scatter(m, v, c=color, s=0.2, linewidths=0.0)
        plt.gca().set_facecolor((0.5,0.5,0.5))
        plt.show()
    return selected