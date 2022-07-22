# %%
import os
import sys

import numpy as np
import pandas as pd
import pyranges as pr
import scipy.stats as ss
from statsmodels.stats.multitest import fdrcorrection

sys.path.append("./")
import matplotlib.pyplot as plt
from lib.utils import matrix_utils
from settings import params, paths

consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensuses = consensuses[[0,1,2]]
consensuses["Name"] = np.arange(len(consensuses))
consensuses.columns = ["Chromosome", "Start", "End", "Name"]
consensusesPr = pr.PyRanges(consensuses)
clusts = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt").astype(int)
# %%
from lib.pyGREAT_normal import pyGREAT as pyGREATglm

enricherglm = pyGREATglm(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
# %%
def symlog_bins(arr, n_bins, a, b, zero_eps=0.1, padding=0):
    """
    Splits a data range into log-like bins but with 0 and negative values taken into account.
    Can be used together with matplotlib 'symlog' axis sacale (i.e. ax.set_xscale('symlog'))
    Feel free to contribute: https://gist.github.com/artoby/0bcf790cfebed5805fbbb6a9853fe5d5
    """
        
    if a > b:
        a, b = b, a
        
    neg_range_log = None
    if a < -zero_eps:
        neg_range_log = [np.log10(-a), np.log10(zero_eps)]
    
    # Add a value to zero bin edges in case a lies within [-zero_eps; zero_eps) - so an additional bin will be added before positive range
    zero_bin_edges = []
    if -zero_eps <= a < zero_eps:
        zero_bin_edges = [a]
            
    pos_range_log = None
    if b > zero_eps:
        pos_range_log = [np.log10(max(a, zero_eps)), np.log10(b)]

    nonzero_n_bin_edges = n_bins + 1 - len(zero_bin_edges)
    
    neg_range_log_size = (neg_range_log[0] - neg_range_log[1]) if neg_range_log is not None else 0
    pos_range_log_size = (pos_range_log[1] - pos_range_log[0]) if pos_range_log is not None else 0
    
    range_log_size = neg_range_log_size + pos_range_log_size
    pos_n_bin_edges_raw = int(round(nonzero_n_bin_edges * (pos_range_log_size/range_log_size))) if range_log_size > 0 else 0
    # Ensure each range has at least 2 edges if it's not empty
    neg_n_bin_edges = max(2, nonzero_n_bin_edges - pos_n_bin_edges_raw) if neg_range_log_size > 0 else 0
    pos_n_bin_edges = max(2, nonzero_n_bin_edges - neg_n_bin_edges) if pos_range_log_size > 0 else 0
    
    neg_bin_edges = []
    if neg_n_bin_edges > 0:
        neg_bin_edges = list(-np.logspace(neg_range_log[0], neg_range_log[1], neg_n_bin_edges))
        
    pos_bin_edges = []
    if pos_n_bin_edges > 0:
        pos_bin_edges = list(np.logspace(pos_range_log[0], pos_range_log[1], pos_n_bin_edges))
    
    result = neg_bin_edges + zero_bin_edges + pos_bin_edges
    return result
class distPlotter:
    def __init__(self, gencode, query, xmin, xmax, extclip=10000):
        self.xmin = xmin
        self.xmax = xmax
        gencodeChr = dict(tuple(gencode.groupby("Chromosome")))
        consensusesCpy = query.copy()
        consensusesCpy["Center"] = consensusesCpy["End"] * 0.5 + consensusesCpy["Start"] * 0.5
        allDists = []
        geneSizes = []
        for i, r in consensusesCpy.iterrows():
            try:
                chrom = gencodeChr[r["Chromosome"]][["Start", "End", "Strand"]].values
                diff = (chrom[:, [0,1]] - r["Center"])
                diff = np.where((chrom[:, 2] == "+")[:, None], diff, diff[:,::-1])
                diff = diff[(diff.max(axis=1) > xmin-extclip) & (diff.min(axis=1) < xmax+extclip)]
                allDists.append(diff)
            except:
                continue
        self.allDists = np.concatenate(allDists)

    def plot(self, tss_size=10000, log=False):
        positions = np.where(self.allDists[:, 0] <= 0, self.allDists[:, 0], 
                                                  self.allDists[:, 1] + tss_size).astype("float")
        if not log:
            plt.hist(positions,np.linspace(self.xmin, self.xmax,200), density=True)
        else:
            bins = symlog_bins(positions,200, a=self.xmin, b=self.xmax)
            plt.hist(positions,bins, density=True)
            plt.xscale("symlog")
        plt.xlabel(f"Distance to TSS/TES (gene length={tss_size}bp)")
        return self
# %%
'''
query = pr.read_bed(paths.outputDir + "rnaseq/Survival2/globally_prognostic.bed")
enrichs = enricherglm.findEnriched(query, consensusesPr)
enricherglm.clusterTreemap(enrichs, output="test.pdf")
'''
try:
    os.mkdir(paths.outputDir + "dist_to_genes/")
except KeyboardInterrupt:
    pass
# %%
gencode = enricherglm.txList
dst = distPlotter(gencode, consensuses,-1e5,1e5)
plt.figure(dpi=500)
dst.plot(10000)
plt.title("All Interg Pol II")
plt.savefig(paths.outputDir + "dist_to_genes/genedist.pdf")
plt.show()
plt.close()
# %%
dst3 = distPlotter(gencode, consensuses,-25e3,35e3)
plt.figure(dpi=500)
dst3.plot(10000)
plt.title("All Interg Pol II")
plt.savefig(paths.outputDir + "dist_to_genes/genedist_25.pdf")
plt.show()
plt.close()
# %%
plt.figure(dpi=500)
dst.plot(10, log=True)
plt.title("All Interg Pol II")
plt.savefig(paths.outputDir + "dist_to_genes/genedist_log_100.pdf")
plt.show()
plt.close()
# %%
test = pr.read_bed(paths.outputDir + "rnaseq/TumorVsNormal2/globally_DE.bed").as_df()
plt.figure(dpi=500)
dst2 = distPlotter(gencode, test, -1e5,1e5).plot(10000)
plt.title("DE 'pan-cancer' Pol II")
plt.savefig(paths.outputDir + "dist_to_genes/genedist_DEcancer.pdf")
plt.show()
plt.close()
# %%
from scipy.stats import kstest

kstest(np.where(dst.allDists[:, 0] <= 0, dst.allDists[:, 0], dst.allDists[:, 1] + 10000),
       np.where(dst2.allDists[:, 0] <= 0, dst2.allDists[:, 0], dst2.allDists[:, 1] + 10000))
# %%
# KDE plot
df = np.where(dst.allDists[:, 0] <= 0, dst.allDists[:, 0], dst.allDists[:, 1] + 10000)
df = pd.DataFrame(df, columns = ["dists"])
df["subset"] = "All"
df2 = np.where(dst2.allDists[:, 0] <= 0, dst2.allDists[:, 0], dst2.allDists[:, 1] + 10000)
df2 = pd.DataFrame(df2, columns = ["dists"])
df2["subset"] = "DE pan-cancer"
df = pd.concat([df, df2])
df.index = np.arange(len(df))
df = df[np.abs(df["dists"]) < 1.25e5]
# %%
import seaborn as sns

plt.figure(dpi=500)
sns.kdeplot(data=df, bw=0.02, x="dists", hue="subset", clip=(-1e5, 1e5), common_norm=False)
plt.title("DE 'pan-cancer' Pol II")
plt.xlabel("Distance to TSS/TES (gene length=10kb)")
plt.savefig(paths.outputDir + "dist_to_genes/genedistKDE_DEcancer.pdf")
plt.show()
plt.close()
# %%
# Tag all tail of genes
tested = [5000,7000,9000]
for tes_ext in tested:
    tails = gencode.copy()

    tesP = tails.loc[tails["Strand"] == "+"]["End"]
    tails.loc[tails["Strand"] == "+", ["Start", "End"]] = np.array([tesP, tesP+tes_ext]).T

    tesM = tails.loc[tails["Strand"] == "-"]["Start"]
    tails.loc[tails["Strand"] == "-", ["Start", "End"]] = np.array([tesM-10000, tesM]).T
    tailedPol2 = consensusesPr.join(pr.PyRanges(tails), False, apply_strand_suffix=False).as_df()
    tailedPol2.drop(["Start_b", "End_b", "Strand"], axis=1, inplace=True)
    uniqueGenes = tailedPol2["gene_name"].unique()
    tailedPol2.to_csv(paths.outputDir + f"dist_to_genes/pol2_{tes_ext}_TES_ext.bed", sep="\t", index=False)
    pd.DataFrame(uniqueGenes).to_csv(paths.outputDir + f"dist_to_genes/genes_{len(uniqueGenes)}_unique_{tes_ext}_TES_ext.csv", index=None, columns=None)
    
# %%
