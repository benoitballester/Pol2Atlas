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
try:
    os.mkdir(paths.outputDir + "rnaseq/count_distrib/")
except FileExistsError:
    pass
palette = sns.color_palette()
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
from scipy.stats import wilcoxon
# Plot average percentage of total reads per kb
pctReadsBG_ENCODE = np.mean(bgCounts/allReads[:, None], axis=1)
pctReadsPol2_ENCODE = np.mean(allCounts/allReads[:, None], axis=1)
dfPctReads = pd.DataFrame(data=np.concatenate([pctReadsBG_ENCODE, pctReadsPol2_ENCODE])*1e6,
                         columns=["Reads per 10m mapped reads per kb"])
dfPctReads["Regions"] = ["ENCODE, control"]*len(pctReadsBG_ENCODE) + ["ENCODE, Pol II Interg"]*len(pctReadsPol2_ENCODE)                         
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(x="Reads per 10m mapped reads per kb", y="Regions", palette=palette, data=dfPctReads, showfliers=False)
sns.stripplot(x="Reads per 10m mapped reads per kb", y="Regions", palette=palette, data=dfPctReads, jitter=0.33, dodge=True, 
                edgecolor="black",alpha=1.0, s=2, linewidth=0.1)
p = wilcoxon(pctReadsBG_ENCODE, pctReadsPol2_ENCODE)
supp = np.sum(pctReadsPol2_ENCODE > pctReadsBG_ENCODE)
medianFC = np.median(pctReadsPol2_ENCODE/pctReadsBG_ENCODE)
plt.title(f"More signal in {supp} / {len(pctReadsPol2_ENCODE)} samples\nMedian fold change:{medianFC}\np={p[1]}")
plt.savefig(paths.outputDir + "rnaseq/count_distrib/readsPerMappedreads.pdf", bbox_inches="tight")
plt.show()
# %%
# Plot count sparsity (% of 0 counts)
normReadsBG = bgCounts/allReads[:, None]
normReadsPolII = allCounts/allReads[:, None]
cutoffs = [0.0, 1e-7, 1e-6, 1e-5]
labels = ["", "per 10 million", "per 1 million", "per 100 000"]
sparsityBG_ENCODE = np.concatenate([np.mean(normReadsBG > c, axis=1) for c in cutoffs])
sparsityPol2_ENCODE = np.concatenate([np.mean(normReadsPolII > c, axis=1) for c in cutoffs])
labelsBG = np.array([[f"1 read {l} mapped reads"] * len(normReadsBG) for l in labels]).ravel()
labelsPolII = np.array([[f"1 read {l} mapped reads"] * len(normReadsPolII) for l in labels]).ravel()
dfSparsity = pd.DataFrame(data=np.concatenate([sparsityBG_ENCODE, sparsityPol2_ENCODE]),
                         columns=["Fraction of non zero counts"])
dfSparsity["Threshold"] = np.concatenate([labelsBG, labelsPolII])
dfSparsity["Regions"] = ["ENCODE, control"]*len(sparsityBG_ENCODE) + ["ENCODE, Pol II Interg"]*len(sparsityPol2_ENCODE)     
plt.figure(dpi=500)
g = sns.FacetGrid(dfSparsity, col="Threshold", sharex=False, height=4, aspect=4/3, col_wrap=2)
g.map_dataframe(sns.boxplot,x="Fraction of non zero counts", y="Regions", palette=palette, showfliers=False)
g.map_dataframe(sns.stripplot,x="Fraction of non zero counts", y="Regions", palette=palette, dodge=True, 
                edgecolor="black", jitter=1/3, alpha=1.0, s=2, linewidth=0.1)
p = wilcoxon(sparsityBG_ENCODE, sparsityPol2_ENCODE)
supp = np.sum(sparsityBG_ENCODE < sparsityPol2_ENCODE)
medianFC = np.median(sparsityPol2_ENCODE/sparsityBG_ENCODE)
# plt.title(f"Less sparsity in {supp} / {len(sparsityPol2_ENCODE)} samples\nMedian fold change:{medianFC}\np={p[1]},")
plt.savefig(paths.outputDir + "rnaseq/count_distrib/sparsity.pdf", bbox_inches="tight")
plt.show()
# %%
from scipy.stats import mannwhitneyu
# Plot non zero cpm read distribution
cpmBg = (bgCounts/allReads[:, None]*1e6).ravel()
cpmPol2 = (allCounts/allReads[:, None]*1e6).ravel()

nzBg = cpmBg[cpmBg > 1e-15]
nzPolII = cpmPol2[cpmPol2 > 1e-15]
largestPct = np.maximum(np.max(nzBg), np.max(nzPolII))
plt.figure(figsize=(6,4), dpi=500)
bins = np.linspace(0, largestPct, 20)
plt.hist(nzPolII, bins, density=True)
plt.hist(nzBg, bins, alpha=0.5, density=True)
plt.yscale("log")
plt.xlabel("CPM")
plt.ylabel("Density (log)")
p = mannwhitneyu(nzBg, nzPolII)
plt.title(f"Distribution of non-zero counts, \np={p[1]}")
plt.legend(["Pol II intergenic", "Control"])
plt.savefig(paths.outputDir + "rnaseq/count_distrib/nonzerocountDistrib.pdf", bbox_inches="tight")
plt.show()
# %%
countsNz = pd.DataFrame()
countsNz["logCPM"] = np.log10(1+1e6*normReadsPolII).ravel()
countsNz["CPM"] = 1e6*normReadsPolII.ravel()
countsNz["Samples"] = np.repeat(np.arange(len(allCounts)), len(allCounts.T))
# %%
sns.boxplot(x="logCPM", y="Samples", orient="h", data=countsNz[countsNz["Samples"] < 1], showfliers=False)
# %%
nzVals = countsNz[countsNz["CPM"] > 1e-15]
nzVals.rename(columns={"logCPM": "Non-zero log(1+CPM)"}, inplace=True)
plt.figure(figsize=(4,35), dpi=500)
sns.boxplot(x="Non-zero log(1+CPM)", y="Samples", orient="h", data=nzVals, showfliers=False)
plt.yticks([], [])
plt.savefig(paths.outputDir + "rnaseq/count_distrib/boxplot_allSamples_no_outliers.pdf", bbox_inches="tight")
plt.show()
plt.close()
# %%
plt.figure(figsize=(4,35), dpi=500)
sns.boxplot(x="Non-zero log(1+CPM)", y="Samples", orient="h", data=nzVals)
plt.yticks([], [])
plt.savefig(paths.outputDir + "rnaseq/count_distrib/boxplot_allSamples_outliers.pdf", bbox_inches="tight")
plt.show()
plt.close()
# %%
