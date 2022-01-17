# %%
# Plot read signal density for TCGA Poly-A RNA-seq, ENCODE Total RNA-seq, 10X 3' Poly-A scRNA-seq
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib import rnaseqFuncs
from lib.utils import plot_utils, matrix_utils
from matplotlib.patches import Patch
from scipy.stats import rankdata, chi2
from scipy.stats import chi2
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection
countDir = "/scratch/pdelangen/projet_these/outputPol2/rnaseq/encode_counts/"
# %%
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/encode_total_rnaseq_annot_0.tsv", 
                        sep="\t", index_col=0)
dlFiles = os.listdir(countDir + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
for f in dlFiles:
    try:
        id = f.split(".")[0]
        countsBG.append(pd.read_csv(countDir + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except Exception as e:
        print(e)
        if len(order) < len(allReads):
            allReads.pop(-1)
        if len(allReads) < len(counts):
            counts.pop(-1)
        if len(counts) < len(countsBG):
            countsBG.pop(-1)
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
bgCounts = np.concatenate(countsBG, axis=1).T
ann, eq = pd.factorize(annotation.loc[order]["Annotation"])
# %%
def ridgePlot(df):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    cols = df.columns
    # Initialize the FacetGrid object
    pal = sns.color_palette("Paired")
    g = sns.FacetGrid(df, row=cols[1], hue=cols[1], aspect=15, height=.5, palette=pal)
    clipVals = np.percentile(df[cols[0]],(0,99))
    # Draw the densities in a few steps
    g.map(sns.kdeplot, cols[0],
        bw_adjust=.5, clip_on=False,
        fill=True, alpha=1, linewidth=1.5, clip=clipVals)
    g.map(sns.kdeplot, cols[0], clip_on=False, color="w", lw=2, bw_adjust=.5, clip=clipVals)
    
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.2, 0.2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, cols[0])

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
# %%
# Plot average percentage of total reads per kb
pctReadsBG_ENCODE = np.mean(bgCounts/allReads[:, None], axis=1)
pctReadsPol2_ENCODE = np.mean(allCounts/allReads[:, None], axis=1)
dfPctReads = pd.DataFrame(data=np.concatenate([pctReadsBG_ENCODE, pctReadsPol2_ENCODE])*1e6,
                         columns=["FPKM per kb"])
dfPctReads["Regions"] = ["ENCODE, control"]*len(pctReadsBG_ENCODE) + ["ENCODE, Pol II Interg"]*len(pctReadsPol2_ENCODE)                         
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(x="FPKM per kb", y="Regions", data=dfPctReads)
# %%
# Plot count sparsity (% of 0 counts)
sparsityBG_ENCODE = np.mean(bgCounts < 0.5, axis=1)
sparsityPol2_ENCODE = np.mean(allCounts < 0.5, axis=1)
dfSparsity = pd.DataFrame(data=np.concatenate([sparsityBG_ENCODE, sparsityPol2_ENCODE]),
                         columns=["Fraction of zero counts"])
dfSparsity["Regions"] = ["ENCODE, control"]*len(sparsityBG_ENCODE) + ["ENCODE, Pol II Interg"]*len(sparsityPol2_ENCODE)     
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(x="Fraction of zero counts", y="Regions", data=dfSparsity)

# %%
# Plot non zero fpkm read distribution
fpkmBg = (bgCounts/allReads[:, None]*1e6).ravel()
fpkmPol2 = (allCounts/allReads[:, None]*1e6).ravel()

nzBg = fpkmBg[fpkmBg > 1e-15]
nzPolII = fpkmPol2[fpkmPol2 > 1e-15]
# %%
largestPct = np.maximum(np.max(nzBg), np.max(nzPolII))
plt.figure(figsize=(6,4), dpi=500)
bins = np.linspace(0, largestPct, 20)
plt.hist(nzPolII, bins, density=True)
plt.hist(nzBg, bins, alpha=0.5, density=True)
plt.yscale("log")
plt.xlabel("FPKM count")
plt.ylabel("Density (log)")
plt.title("Distribution of non-zero counts")
plt.legend(["Pol II intergenic", "Control"])
# %%
