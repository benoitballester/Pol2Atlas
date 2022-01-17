import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib.utils import plot_utils, matrix_utils
from matplotlib.patches import Patch
from scipy.stats import rankdata, chi2
from scipy.special import erfinv
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection
import scipy.interpolate as si


def filterDetectableGenes(counts, readMin, expMin):
    return np.sum(counts >= readMin, axis=0) >= expMin


def variableSelection(matrix, alpha=0.05, percentile=5, plot=False):
    # Estimate mean and variance for each feature
    m = np.mean(matrix, axis=0)
    v = np.var(matrix, axis=0)
    order = np.argsort(m)
    sortedMean = m[order]
    sortedVar = v[order]
    evalPts = 100
    # Compute a regularized estimate of the variance according to the mean
    means = np.zeros(evalPts+1)
    var5Pct = np.zeros(evalPts+1)
    for i in range(evalPts + 1):
        valMin = max(0, int(len(m) / evalPts * (i-0.5)))
        valMax = min(len(m), int(len(m) / evalPts * (i+0.5)))
        means[i] = sortedMean[np.clip(int(len(m) / evalPts * (i)),0,len(sortedMean)-1)]
        var5Pct[i] = np.percentile(sortedVar[valMin:valMax], percentile)
    means[0] = sortedMean[0]
    var5Pct[0] = sortedVar[0]
    means[-1] = sortedMean[-1]
    var5Pct[-1] = sortedVar[-1]
    fittedSD = si.interp1d(means, var5Pct)(m)
    # Compute normal distribution scaled deviance 
    stat = np.square((matrix - m))/fittedSD
    deviance = np.sum(stat, axis=0)
    # Deviance follows a chi square distribution under null hypothesis
    pvals = chi2.sf(deviance, len(matrix)+1)
    # Select significantly poorly fitted features 
    selected = fdrcorrection(pvals, method="negcorr")[1] < alpha
    if plot:
        pct = rankdata(deviance)/len(deviance)
        color = sns.color_palette("magma", as_cmap=True)(pct)
        plt.figure(dpi=500)
        plt.scatter(m, v, c=color, s=0.2, linewidths=0.0)
        plt.gca().set_facecolor((0.5,0.5,0.5))
        # plt.plot(means, var5Pct)
        plt.scatter(m, fittedSD, s=1.0, linewidths=0.0)
        plt.xlabel("Mean rank")
        plt.ylabel("Mean rank variance")
        plt.show()
        plt.figure(dpi=500)
        color = np.array([[0,0,1]]*len(m))
        color[selected] = [1,0,0]
        plt.scatter(m, v, c=color, s=0.2, linewidths=0.0)
        plt.gca().set_facecolor((0.5,0.5,0.5))
        plt.xlabel("Mean rank")
        plt.ylabel("Mean rank variance")
    return selected


def binomDevianceSelection(counts, alpha=0.05, plot=False):
    # Deviance based feature selection
    from kneed import KneeLocator
    n_i = np.sum(counts, axis=1)[:, None]
    countsProp = counts/n_i
    pi_j = np.mean(countsProp, axis=0)
    v = counts * np.log(1e-15 + counts / (n_i*pi_j)) + (n_i - counts) * np.log(1e-15 + (n_i - counts)/(n_i * (1-pi_j)))
    deviance = np.sum(v, axis=0)
    orderedDev = np.argsort(deviance)[::-1]
    kneedl = KneeLocator(np.arange(len(deviance)), deviance[orderedDev],
                        direction="decreasing", curve="convex", online=True)
    bestR = kneedl.knee
    kneedl.plot_knee()
    selected = orderedDev[:bestR]
    return selected


def quantileTransform(counts):
    rg = ((rankdata(counts, axis=0)-0.5)/counts.shape[0])*2.0 - 1.0
    return erfinv(rg)

