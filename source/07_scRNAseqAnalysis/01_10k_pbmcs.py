# %%
import os
from lib.utils.reusableUtest import mannWhitneyAsymp
import fastcluster
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scanpy as sc
import scipy.stats as ss
import seaborn as sns
import sklearn.metrics as metrics
import umap
from lib import rnaseqFuncs
from lib.utils import matrix_utils, plot_utils
from matplotlib.patches import Patch
from scipy.io import mmread
from scipy.spatial.distance import dice
from scipy.stats import chi2, rankdata
from settings import params, paths
from statsmodels.stats.multitest import fdrcorrection

try:
    os.mkdir(paths.outputDir + "scrnaseq/")
except FileExistsError:
    pass
try:
    os.mkdir(paths.outputDir + "scrnaseq/DE/")
except FileExistsError:
    pass
try:
    os.mkdir(paths.outputDir + "scrnaseq/markerGenes/")
except FileExistsError:
    pass

# %%
# Scanpy params
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
results_file = paths.tempDir + "pbmc10k.h5ad"
sc.settings.figdir = ""
# %%
# Run pipeline on gene expression data
# Load data with scanpy
adata = sc.read_10x_mtx(
    paths.pbmc10k + "10k_PBMC_3p_nextgem_Chromium_X_raw_feature_bc_matrix/filtered_feature_bc_matrix/",  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)  

sc.pl.highest_expr_genes(adata, n_top=20, )
# QC
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
plt.savefig(paths.outputDir + "scrnaseq/qc_1.pdf")
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
plt.savefig(paths.outputDir + "scrnaseq/qc_2.pdf")
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
plt.savefig(paths.outputDir + "scrnaseq/qc_3.pdf")
print("Found %i cells before filtering"%len(adata))
# Filter cells with too many genes, not enough genes or too many counts on mt genes
adata = adata[adata.obs.n_genes_by_counts < 4000, :]
adata = adata[adata.obs.total_counts > 1000, :]
adata = adata[adata.obs.pct_counts_mt < 13, :]
print("Found %i cells after filtering"%len(adata))
counts = np.array(adata.X.todense().astype("int32"))

nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=2)
counts = counts[:, nzCounts]

sf = np.sum(counts, axis=1)
sf = sf/sf.mean()

countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf, residuals="deviance")
hv = countModel.hv
feat = countModel.residuals
decomp = rnaseqFuncs.permutationPA_PCA(feat, 3, max_rank=100, returnModel=False, whiten=True)

embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0, low_memory=False, metric="correlation").fit_transform(decomp)

import matplotlib.pyplot as plt
# Plot UMAP with read information
plt.figure(dpi=500)
readsPerCell = np.sum(counts, axis=1)
readsPerCell = readsPerCell-np.min(readsPerCell)
plt.scatter(embedding[:, 0], embedding[:, 1], s=1.0, linewidths=0, c=np.sqrt(readsPerCell/readsPerCell.max()))
plt.show()
# Cluster
labelsGenes = matrix_utils.graphClustering(decomp, metric="correlation", approx=False, restarts=10, snn=True).astype("str")
colors = sns.color_palette("tab20")
palettePlotly = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
colormap = dict(zip(np.unique(labelsGenes), palettePlotly))
df = pd.DataFrame({"x":embedding[:, 0] ,"y":embedding[:, 1], "cluster":labelsGenes})
fig = px.scatter(df, x="x", y="y", color="cluster", color_discrete_map=colormap,
                hover_data=['cluster'], width=800, height=800)
fig.update_traces(marker=dict(size=3.0))
fig.write_image(paths.outputDir + "scrnaseq/10k_pbmc_genes.pdf")
fig.write_html(paths.outputDir + "scrnaseq/10k_pbmc_genes.pdf" + ".html")
fig.show()
# %%
# DE genes
# tester = mannWhitneyAsymp(countModel.residuals)
pctThreshold = 0.1
lfcMin = 0.25
matchinglabels = labelsGenes
resClust = dict()
for i in np.unique(matchinglabels):
    print(i)
    grp = (matchinglabels == i).astype(int)
    res2 = ss.ttest_ind(countModel.residuals[grp == 1], countModel.residuals[grp != 1], axis=0,
                    alternative="greater")
    sig = fdrcorrection(res2[1])[0]
    minpct = np.mean(counts[matchinglabels == i] > 0.5, axis=0) > max(0.1, 1.5/grp.sum())
    fc = np.mean(countModel.normed[matchinglabels == i], axis=0) / (1e-9+np.mean(countModel.normed[matchinglabels != i], axis=0))
    lfc = np.log2(fc) > lfcMin
    sig = sig & lfc & minpct
    print(sig.sum())
    delta = np.mean(countModel.residuals[matchinglabels == i], axis=0) - (np.mean(countModel.residuals[matchinglabels != i], axis=0))
    res = pd.DataFrame(res2[::-1], columns=adata.var.index[nzCounts], index=["pval", "stat"]).T
    res["Upreg"] = 1-sig.astype(int)
    res["fc"] = -delta
    order = np.lexsort(res[["fc","pval","Upreg"]].values.T)
    res["fc"] = delta
    res["Upreg"] = sig.astype(int)
    res = res.iloc[order]
    resClust[i] = res
    res.to_csv(paths.outputDir + f"scrnaseq/DE/res_genes_{i}.csv")
    test = pd.Series(adata.var.index[nzCounts][order][sig[order]])
    test.to_csv(paths.outputDir + f"scrnaseq/DE/genes_{i}.csv", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
# %%
# Plot top markers
for clustID in np.unique(labelsGenes):
    gridSize = (4,4)
    fig, axs = plt.subplots(gridSize[0], gridSize[1], dpi=500)
    [axi.set_axis_off() for axi in axs.ravel()]
    axs[0][0].scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=0.5,linewidths=0, c=clustID==labelsGenes)
    axs[0][0].set_title("Cluster " + clustID, fontsize=8)
    for i in range(1, gridSize[0]*gridSize[1]):
        gene = resClust[clustID].index[i-1]
        readsPerCell = countModel.normed[:, adata.var.index[nzCounts] == gene]
        yi = i%gridSize[1]
        xi = int(i/gridSize[0])
        axs[xi][yi].scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=0.5,linewidths=0, c=np.sqrt(readsPerCell/readsPerCell.max()))
        axs[xi][yi].set_title(gene, fontsize=6)
    fig.savefig(paths.outputDir + "scrnaseq/markerGenes/" + clustID + ".png", bbox_inches="tight", dpi=500)
# %%
# Manually curated labels
clustGeneLabels = {"0":"CD14+ Monocytes",
                   "1":"Naive CD4 T-Cells",
                   "2":"Naive CD8 T-Cells",
                   "3":"Memory CD4 T-Cells",
                   "4":"CD14+ S100A8+ Monocytes",
                   "5":"CPVL+ Monocytes",
                   "6":"Memory B Cells",
                   "7":"Naive B Cells",
                   "8":"NK Cells",
                   "9":"GZMH+ Memory CD8 T-Cell",
                   "10":"FCGR3A+ Monocytes",
                   "11":"GZMK+ Memory CD8 T-Cells",
                   "12":"MAIT T-Cells",
                   "13":"Regulatory T-Cells",
                   "14":"Plasmacytoid DC",
                   "15":"Myeloid DC",
                   "16":"NK Cells 2",
                   "17":"Platelets",
                   "18":"Plasmablasts"
                   }
lowLevelAnnot = {"0":"Monocytes",
                "1":"T-Cells",
                "2":"T-Cells",
                "3":"T-Cells",
                "4":"Monocytes",
                "5":"Monocytes",
                "6":"B Cells",
                "7":"B Cells",
                "8":"NK Cells",
                "9":"T-Cell",
                "10":"Monocytes",
                "11":"T-Cells",
                "12":"T-Cells",
                "13":"T-Cells",
                "14":"DC",
                "15":"DC",
                "16":"NK Cells",
                "17":"Platelets",
                "18":"B Cells"
                }
annotatedLabels = [clustGeneLabels[i] for i in labelsGenes]
annotatedLowLevel = [lowLevelAnnot[i] for i in labelsGenes]
colors = sns.color_palette("tab20")
palettePlotly = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
colormap = dict(zip(np.unique(annotatedLabels), palettePlotly))
df = pd.DataFrame({"x":embedding[:, 0] ,"y":embedding[:, 1], "cluster":annotatedLabels})
fig = px.scatter(df, x="x", y="y", color="cluster", color_discrete_map=colormap,
                hover_data=['cluster'], width=800, height=800)
fig.update_traces(marker=dict(size=3.0))
fig.write_image(paths.outputDir + "scrnaseq/10k_pbmc_genes_ann.pdf")
fig.write_html(paths.outputDir + "scrnaseq/10k_pbmc_genes_ann.pdf" + ".html")
fig.show()
# %%
# Run pipeline on Pol II probe expression
# For whatever reason scanpy does not want to load our files
polIICounts = mmread(paths.pbmc10k + "Pol II counts/filtered_feature_bc_matrix/matrix.mtx.gz")
polIICounts = np.array(polIICounts.todense()).astype("int32")
barcodes = pd.read_csv(paths.pbmc10k + "Pol II counts/filtered_feature_bc_matrix/barcodes.tsv.gz", header=None).values
features = pd.read_csv(paths.pbmc10k + "Pol II counts/filtered_feature_bc_matrix/features.tsv.gz", sep="\t", header=None).values
# %%
# Make barcodes match between the two analyses
polIICounts = pd.DataFrame(polIICounts.T, index=barcodes.ravel())
foundBarcodes = np.intersect1d(adata.obs.index, polIICounts.index)
polIICounts = polIICounts.loc[foundBarcodes].values
print(len(adata.obs.index), " found barcodes for genes.")
print(len(polIICounts), " found barcodes for Pol II.")
# %%
# Filter not detectable probes
nzCountsPolII = rnaseqFuncs.filterDetectableGenes(polIICounts, readMin=1, expMin=1)
matchingsf = pd.Series(sf, adata.obs.index).loc[foundBarcodes].values
# matchingsf = np.sum(polIICounts, axis=1)
matchingsf = matchingsf/matchingsf.mean()
polIICounts = polIICounts[:, nzCountsPolII]
# %%
# Fit count model
countModelPolII = rnaseqFuncs.RnaSeqModeler().fit(polIICounts, matchingsf, residuals="deviance")
hvPolII = countModelPolII.hv
# %%
# PCA with same number of components as gene analysis
from sklearn.decomposition import PCA
feat = countModelPolII.residuals[:, rnaseqFuncs.filterDetectableGenes(polIICounts, readMin=1, expMin=3)]
decomp = rnaseqFuncs.permutationPA_PCA(feat, 3, max_rank=100, returnModel=False, whiten=True)
# %%
# UMAP
embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0, low_memory=False, metric="correlation").fit_transform(decomp)
import matplotlib.pyplot as plt

plt.figure(dpi=500)
readsPerCell = np.sum(polIICounts, axis=1)
readsPerCell = readsPerCell-np.min(readsPerCell)
plt.scatter(embedding[:, 0], embedding[:, 1],s=1.0,linewidths=0, c=np.sqrt(readsPerCell/readsPerCell.max()))

plt.show()
# %%
# Draw cluster labels on top of umap
colors = sns.color_palette("tab20")
palettePlotly = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
colormap = dict(zip(np.unique(labelsGenes), palettePlotly))
df = pd.DataFrame({"cluster":labelsGenes})
df.index = adata.obs.index
df = df.loc[foundBarcodes]
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]
fig = px.scatter(df, x="x", y="y", color="cluster", color_discrete_map=colormap,
                hover_data=['cluster'], width=800, height=800)
fig.update_traces(marker=dict(size=3.0))
fig.write_image(paths.outputDir + "scrnaseq/10k_pbmc_pol2probes_clustergenes.pdf")
fig.write_html(paths.outputDir + "scrnaseq/10k_pbmc_pol2probes_clustergenes.pdf" + ".html")
fig.show()
# %%
# Cluster Pol II expr wise
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
matchinglabels = pd.Series(labelsGenes, adata.obs.index).loc[foundBarcodes].values
labels = matrix_utils.graphClustering(decomp, metric="correlation", approx=False, restarts=10, snn=True).astype("str")
colors = sns.color_palette("tab20")
print(adjusted_rand_score(matchinglabels, labels), adjusted_mutual_info_score(matchinglabels, labels))
palettePlotly = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
colormap = dict(zip(np.unique(labels), palettePlotly))
df = pd.DataFrame({"x":embedding[:, 0] ,"y":embedding[:, 1], "cluster":labels})
fig = px.scatter(df, x="x", y="y", color="cluster", color_discrete_map=colormap,
                hover_data=['cluster'], width=800, height=800, 
                title=f"ARI : {adjusted_rand_score(matchinglabels, labels)}\nAMI : {adjusted_mutual_info_score(matchinglabels, labels)}")
fig.update_traces(marker=dict(size=3.0))
fig.write_image(paths.outputDir + "scrnaseq/10k_pbmc_pol2probes_clusterPol2.pdf")
fig.write_html(paths.outputDir + "scrnaseq/10k_pbmc_pol2probes_clusterPol2.pdf" + ".html")
fig.show()
# %%
# Setup DE analysis
import pyranges as pr

# tester = mannWhitneyAsymp(countModelPolII.residuals)
consensuses = pr.read_gtf(paths.tempDir + "Pol2.gtf").as_df()
consensuses.drop(["Source", "Feature", "Frame", "gene_biotype", "transcript_id", "gene_name"], 1, inplace=True)
consensuses.columns = ["Chromosome", "Start", "End","Score","Strand","Name"]
consensuses = consensuses[["Chromosome", "Start", "End", "Name", "Strand"]]
consensuses.index = consensuses["Name"].astype("int")
consensuses = consensuses.loc[features[:, 1]]
# %%
# DE Pol II
pctThreshold = 0.1
lfcMin = 0.25
resClustP2 = dict()
for i in np.unique(matchinglabels):
    print(i)
    grp = (matchinglabels == i).astype(int)
    res2 = ss.ttest_ind(countModelPolII.residuals[grp == 1], countModelPolII.residuals[grp == 0], axis=0,
                    alternative="greater", equal_var=False)
    sig = fdrcorrection(res2[1])[0]
    minpct = np.mean(polIICounts[matchinglabels == i] > 0.5, axis=0) > max(0.1, 1.5/grp.sum())
    fc = np.mean(countModelPolII.normed[matchinglabels == i], axis=0) / (1e-9+np.mean(countModelPolII.normed[matchinglabels != i], axis=0))
    lfc = np.log2(fc) > lfcMin
    sig = sig & lfc & minpct
    print(sig.sum())
    delta = np.mean(countModelPolII.residuals[matchinglabels == i], axis=0) - (np.mean(countModelPolII.residuals[matchinglabels != i], axis=0))
    res = pd.DataFrame(res2[::-1], columns=consensuses.index[nzCountsPolII], index=["pval", "stat"]).T
    res["Upreg"] = 1-sig.astype(int)
    res["fc"] = -delta
    res["order2"] = np.arange(len(res))
    order = np.lexsort(res[["fc","pval","Upreg"]].values.T)
    res["fc"] = delta
    res["Upreg"] = sig.astype(int)
    res = res.iloc[order]
    resClustP2[i] = res
    res.to_csv(paths.outputDir + f"scrnaseq/DE/res_{i}.csv")
    test = consensuses[nzCountsPolII][sig]
    test.to_csv(paths.outputDir + f"scrnaseq/DE/bed_{i}.bed", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
# %%
# Plot top markers for Pol II probes
for clustID in np.unique(matchinglabels):
    gridSize = (4,4)
    fig, axs = plt.subplots(gridSize[0], gridSize[1], dpi=500)
    [axi.set_axis_off() for axi in axs.ravel()]
    axs[0][0].scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=0.5,linewidths=0, c=clustID==matchinglabels)
    axs[0][0].set_title("Cluster " + clustID, fontsize=8)
    for i in range(1, gridSize[0]*gridSize[1]):
        gene = resClustP2[clustID]["order2"].iloc[i-1]
        geneId2 = resClustP2[clustID].index[i-1]
        isSig = resClustP2[clustID]["Upreg"].iloc[i-1]
        if isSig == 0:
            continue
        readsPerCell = countModelPolII.normed[:, gene]
        yi = i%gridSize[1]
        xi = int(i/gridSize[0])
        axs[xi][yi].scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=0.5, linewidths=0, c=np.sqrt(readsPerCell/readsPerCell.max()))
        axs[xi][yi].set_title(geneId2, fontsize=6)
    fig.savefig(paths.outputDir + "scrnaseq/markerGenes/PolII_" + clustID + ".png", bbox_inches="tight", dpi=500)

# %%
