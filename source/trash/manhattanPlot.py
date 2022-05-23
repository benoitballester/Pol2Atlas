# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
from statsmodels.stats.multitest import fdrcorrection
import sys
sys.path.append("./")
from settings import params, paths



chrFile = pd.read_csv(paths.genomeFile, sep="\t", index_col=0, header=None)
sortedIdx = ["chr1", 'chr2','chr3','chr4','chr5','chr6',
              'chr7','chr8','chr9', 'chr10', 'chr11','chr12','chr13','chr14','chr15','chr16','chr17',
              'chr18','chr19','chr20','chr21','chr22','chrX','chrY']
chrFile = chrFile.loc[sortedIdx]
bedFile = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
statFile = pd.read_csv(paths.outputDir + "rnaseq/TumorVsNormal2/TCGA-KIRC/stats.csv", sep="\t", index_col=0)
bedFile = bedFile.loc[statFile.index]
# %%
def manhattanPlot(coords, chrInfo, pvalues, es, maxLogP=30, threshold="fdr", fdrThreshold=0.05):
    fig, ax = plt.subplots(dpi=500)
    fractPos = (chrInfo.values.ravel()/np.sum(chrInfo.values).ravel())
    offsets = np.insert(np.cumsum(fractPos),0,0)
    for i, c in enumerate(chrInfo.index):
        usedIdx = coords[0] == c
        coordSubset = coords[usedIdx]
        x = offsets[i] + (coordSubset[1]*0.5 + coordSubset[2]*0.5)/chrInfo.loc[c].values * fractPos[i]
        y = np.clip(-np.log10(pvalues[usedIdx]),0, maxLogP)
        ax.scatter(x,y, s=1.0, linewidths=0)
    ax.set_xticks(offsets[:-1]+0.5*fractPos, chrInfo.index, rotation=90, fontsize=8)
    if threshold is not None:
        if threshold == "fdr":
            sortedP = np.sort(pvalues)[::-1]
            fdrSig = np.searchsorted(fdrcorrection(np.sort(pvalues)[::-1], fdrThreshold)[0], True)
            if fdrSig > 0:
                threshold = -np.log10(sortedP[fdrSig])
            else:
                threshold = -np.log10(0.05/len(coords))
        ax.set_xlim(-0.02,1.02)
        ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], color=(0,0,0), linestyles="dashed")
            # plt.text(plt.xlim()[0], threshold*1.1, f"{fdrThreshold} FDR", fontsize=8)
    ax.set_ylabel("-log10(p-value)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax

fig, ax = manhattanPlot(bedFile, chrFile, statFile["p"], statFile["coef"])

# %%
