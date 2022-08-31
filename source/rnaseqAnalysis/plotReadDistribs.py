# %%
# Plot read signal density for TCGA Poly-A RNA-seq, ENCODE Total RNA-seq, 10X 3' Poly-A scRNA-seq
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("./")
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
from scipy.stats import wilcoxon
from scipy.sparse import csc_array, coo_array, vstack

try:
    os.mkdir(paths.outputDir + "rnaseq/count_distrib/")
except FileExistsError:
    pass
palette = sns.color_palette()
# %%
# %%
def plotReadDistribs(allReads, allCounts, bgCounts, suffix, plotAllCounts=False):
    # Plot average percentage of total reads per kb
    sf = np.sqrt(allReads.shape[0]/510)
    normReadsBG = bgCounts.copy()
    normReadsBG.data = normReadsBG.data / np.take(allReads, normReadsBG.indices)
    normReadsPolII = allCounts.copy()
    normReadsPolII.data = normReadsPolII.data / np.take(allReads, normReadsPolII.indices)
    pctReadsBG_ENCODE = (normReadsBG).mean(axis=1)
    pctReadsPol2_ENCODE = (normReadsPolII).mean(axis=1)
    dfPctReads = pd.DataFrame(data=np.concatenate([pctReadsBG_ENCODE, pctReadsPol2_ENCODE])*1e6,
                            columns=["Reads per 10m mapped reads per kb"])
    dfPctReads["Regions"] = [f"{suffix}, control"]*len(pctReadsBG_ENCODE) + [f"{suffix}, Pol II Interg"]*len(pctReadsPol2_ENCODE)                         
    plt.figure(figsize=(6,4), dpi=500)
    sns.boxplot(x="Reads per 10m mapped reads per kb", y="Regions", palette=palette, data=dfPctReads, showfliers=False)
    sns.stripplot(x="Reads per 10m mapped reads per kb", y="Regions", palette=palette, data=dfPctReads, jitter=0.33, dodge=True, 
                    edgecolor="black",alpha=1.0, s=2/sf, linewidth=0.1)
    p = wilcoxon(pctReadsBG_ENCODE, pctReadsPol2_ENCODE)
    supp = np.sum(pctReadsPol2_ENCODE > pctReadsBG_ENCODE)
    medianFC = np.median(pctReadsPol2_ENCODE/pctReadsBG_ENCODE)
    plt.title(f"More signal in {supp} / {len(pctReadsPol2_ENCODE)} samples\nMedian fold change:{medianFC}\np={p[1]}")
    plt.savefig(paths.outputDir + f"rnaseq/count_distrib/readsPerMappedreads_{suffix}.pdf", bbox_inches="tight")
    plt.show()
    # Plot count sparsity (% of 0 counts)
    print(normReadsBG)
    cutoffs = [0.0, 1e-7, 1e-6, 1e-5]
    labels = ["", "per 10 million", "per 1 million", "per 100 000"]
    sparsityBG_ENCODE = np.concatenate([(normReadsBG > c).mean(axis=1) for c in cutoffs])
    sparsityPol2_ENCODE = np.concatenate([(normReadsPolII > c).mean(axis=1) for c in cutoffs])
    labelsBG = np.array([[f"1 read {l} mapped reads"] * normReadsBG.shape[0] for l in labels]).ravel()
    labelsPolII = np.array([[f"1 read {l} mapped reads"] * normReadsPolII.shape[0] for l in labels]).ravel()
    dfSparsity = pd.DataFrame(data=np.concatenate([sparsityBG_ENCODE, sparsityPol2_ENCODE]),
                            columns=["Fraction of non zero counts"])
    dfSparsity["Threshold"] = np.concatenate([labelsBG, labelsPolII])
    dfSparsity["Regions"] = [f"{suffix}, control"]*len(sparsityBG_ENCODE) + [f"{suffix}, Pol II Interg"]*len(sparsityPol2_ENCODE)     
    plt.figure(dpi=500)
    g = sns.FacetGrid(dfSparsity, col="Threshold", sharex=False, height=4, aspect=4/3, col_wrap=2)
    g.map_dataframe(sns.boxplot,x="Fraction of non zero counts", y="Regions", palette=palette, showfliers=False)
    g.map_dataframe(sns.stripplot,x="Fraction of non zero counts", y="Regions", palette=palette, dodge=True, 
                    edgecolor="black", jitter=1/3, alpha=1.0, s=2/sf, linewidth=0.1)
    p = wilcoxon(sparsityBG_ENCODE, sparsityPol2_ENCODE)
    supp = np.sum(sparsityBG_ENCODE < sparsityPol2_ENCODE)
    medianFC = np.median(sparsityPol2_ENCODE/sparsityBG_ENCODE)
    plt.savefig(paths.outputDir + f"rnaseq/count_distrib/sparsity_{suffix}.pdf", bbox_inches="tight")
    plt.show()
    from scipy.stats import mannwhitneyu
    # Plot non zero cpm read distribution
    if plotAllCounts:
        cpmBg = (normReadsBG*1e6).ravel()
        cpmPol2 = (normReadsPolII*1e6).ravel()
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
        plt.savefig(paths.outputDir + f"rnaseq/count_distrib/nonzerocountDistrib_{suffix}.pdf", bbox_inches="tight")
        plt.show()
        # Plot distrib of each experiment
        countsNz = pd.DataFrame()
        countsNz["logCPM"] = np.log10(1+1e6*normReadsPolII).ravel()
        countsNz["CPM"] = 1e6*normReadsPolII.ravel()
        countsNz["Samples"] = np.repeat(np.arange(len(allCounts)), len(allCounts.T))
        sns.boxplot(x="logCPM", y="Samples", orient="h", data=countsNz[countsNz["Samples"] < 1], showfliers=False)
        nzVals = countsNz[countsNz["CPM"] > 1e-15]
        nzVals.rename(columns={"logCPM": "Non-zero log(1+CPM)"}, inplace=True)
        plt.figure(figsize=(4,35), dpi=500)
        sns.boxplot(x="Non-zero log(1+CPM)", y="Samples", orient="h", data=nzVals, showfliers=False)
        plt.yticks([], [])
        plt.savefig(paths.outputDir + f"rnaseq/count_distrib/boxplot_allSamples_no_outliers_{suffix}.pdf", bbox_inches="tight")
        plt.show()
        plt.close()
        plt.figure(figsize=(4,35), dpi=500)
        sns.boxplot(x="Non-zero log(1+CPM)", y="Samples", orient="h", data=nzVals)
        plt.yticks([], [])
        plt.savefig(paths.outputDir + f"rnaseq/count_distrib/boxplot_allSamples_outliers_{suffix}.pdf", bbox_inches="tight")
        plt.show()
        plt.close()
    # Fraction of intergenic counts on Pol II probes
    pctPerSample = allCounts.sum(axis=1) / (allCounts.sum(axis=1) + bgCounts.sum(axis=1))
    plt.figure(dpi=500)
    sns.boxplot(data=pctPerSample, orient="h", showfliers=False)
    sns.stripplot(data=pctPerSample, jitter=0.33, dodge=True, 
                    edgecolor="black",alpha=1.0, s=2/sf, linewidth=0.1, orient="h")
    plt.xlim((0,1))
    plt.yticks([], [])
    plt.xlabel("Fraction of intergenic reads on Pol II probes")
    plt.text(allCounts.shape[1]/bgCounts.shape[1], plt.ylim()[1]*1.05, "Expected", color="red", ha="center")
    plt.vlines(allCounts.shape[1]/bgCounts.shape[1], plt.ylim()[0], plt.ylim()[1], color="red", linestyles="dashed")
    plt.savefig(paths.outputDir + f"rnaseq/count_distrib/boxplot_pctInterg_{suffix}.pdf", bbox_inches="tight")
    plt.show()
    print("c")
    # Fraction of mapped reads
    pctPerSample = allCounts.sum(axis=1) / allReads[None, :] * 100
    plt.figure(dpi=500)
    sns.boxplot(data=pctPerSample, orient="h", showfliers=False)
    sns.stripplot(data=pctPerSample, jitter=0.33, dodge=True, 
                    edgecolor="black",alpha=1.0, s=2/sf, linewidth=0.1, orient="h")
    plt.yticks([], [])
    plt.xlabel("Percentage of mapped reads in Pol II probes")
    plt.savefig(paths.outputDir + f"rnaseq/count_distrib/boxplot_pctmapped_{suffix}.pdf", bbox_inches="tight")
    plt.show()
    print("a")
    pctPerSample = np.sum(normReadsPolII > 1e-7, axis=1) / (np.sum(normReadsPolII > 1e-7, axis=1) + np.sum(normReadsBG > 1e-7, axis=1))
    # Fraction of mapped reads
    pctPerSample = np.sum(allCounts, axis=1) / allReads[None, :] * 100
    plt.figure(dpi=500)
    sns.boxplot(data=pctPerSample, orient="h", showfliers=False)
    sns.stripplot(data=pctPerSample, jitter=0.33, dodge=True, 
                    edgecolor="black",alpha=1.0, s=2/sf, linewidth=0.1, orient="h")
    plt.xlim((0,5))
    plt.yticks([], [])
    plt.xlabel("Percentage of mapped reads in Pol II probes")
    plt.savefig(paths.outputDir + f"rnaseq/count_distrib/boxplot_pctmapped_fixed_scale_{suffix}.pdf", bbox_inches="tight")
    plt.show()
    print("b")
    pctPerSample = (normReadsPolII > 1e-7).sum(axis=1) / ((normReadsPolII > 1e-7).sum(axis=1) + (normReadsBG > 1e-7).sum(axis=1))
    # Pct of > 1 read per 10m mapped reads
    plt.figure(dpi=500)
    sns.boxplot(data=pctPerSample, orient="h", showfliers=False)
    sns.stripplot(data=pctPerSample, jitter=0.33, dodge=True, 
                    edgecolor="black",alpha=1.0, s=2/np.sqrt(len(allReads)/510), linewidth=0.1, orient="h")
    plt.xlim((0,1))
    plt.yticks([], [])
    plt.xlabel("Fraction of intergenic reads on Pol II probes")
    plt.text(allCounts.shape[1]/bgCounts.shape[1], plt.ylim()[1]*1.05, "Expected", color="red", ha="center")
    plt.vlines(allCounts.shape[1]/bgCounts.shape[1], plt.ylim()[0], plt.ylim()[1], color="red", linestyles="dashed")
    plt.savefig(paths.outputDir + f"rnaseq/count_distrib/boxplot_pct_sparsity_Interg_{suffix}.pdf", bbox_inches="tight")
    plt.show()

# %%
# Plot for encode
annotation = pd.read_csv(paths.encodeAnnot, 
                        sep="\t", index_col=0)
countDir = paths.countsENCODE
dlFiles = os.listdir(countDir + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
for f in dlFiles:
    try:
        id = f.split(".")[0]
        countsBG.append(coo_array(pd.read_csv(countDir + "BG/" + f, header=None, skiprows=2).values).T)
        status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        values = pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values
        counts.append(coo_array(values).T)
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
allCounts = csc_array(vstack(counts))
bgCounts = csc_array(vstack(countsBG))
plotReadDistribs(allReads, allCounts, bgCounts, suffix="ENCODE")
# %%
# Plot TCGA
annotation = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
dlFiles = os.listdir(paths.countsTCGA + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
i = 0
for f in np.array(dlFiles):
    try:
        i += 1
        if i % 100 == 0:
            print(i)
        id = f.split(".")[0]
        countsBG.append(coo_array(pd.read_csv(paths.countsTCGA + "BG/" + f, header=None, skiprows=2).values).T)
        status = pd.read_csv(paths.countsTCGA + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(coo_array(pd.read_csv(paths.countsTCGA + "500centroid/" + f, header=None, skiprows=2).values).T)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except:
        continue
allReads = np.array(allReads)
allCounts = csc_array(vstack(counts))
bgCounts = csc_array(vstack(countsBG))
plotReadDistribs(allReads, allCounts, bgCounts, suffix="TCGA")
# %%
# Plot GTEx
annotation = pd.read_csv(paths.gtexData + "/tsvs/sample.tsv", 
                        sep="\t", index_col="specimen_id")

colors = pd.read_csv(paths.gtexData + "colors.txt", 
                        sep="\t", index_col="tissue_site_detail")
dlFiles = os.listdir(paths.countsGTEx + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
allStatus = []
for f in dlFiles:
    try:
        id = ".".join(f.split(".")[:-3])
        countsBG.append(coo_array(pd.read_csv(paths.countsGTEx + "BG/" + f, header=None, skiprows=2).values).T)
        status = pd.read_csv(paths.countsGTEx + "500centroid/" + id + ".counts.summary",
                                header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(coo_array(pd.read_csv(paths.countsGTEx + "500centroid/" + f, header=None, skiprows=2).values).T)
        allStatus.append(status)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(f.split(".")[0])
    except:
        print(f, "missing")
        continue

ann, eq = pd.factorize(annotation.loc[order]["tissue_type"])
allReads = np.array(allReads)
allCounts = csc_array(vstack(counts))
bgCounts = csc_array(vstack(countsBG))
plotReadDistribs(allReads, allCounts, bgCounts, suffix="GTEx")

# %%
