# %%
import os

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
# %%
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
results_file = paths.tempDir + "pbmc10k.h5ad"
# %%
# Run pipeline on gene expression data
# Load data with scanpy
adata = sc.read_10x_mtx(
    paths.pbmc10k + "10k_PBMC_3p_nextgem_Chromium_X_raw_feature_bc_matrix/filtered_feature_bc_matrix/",  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)  

sc.pl.highest_expr_genes(adata, n_top=20, )

adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)

sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

# Filter cells with too many genes, not enough genes or too many counts on mt genes
adata = adata[adata.obs.n_genes_by_counts < 4000, :]
adata = adata[adata.obs.total_counts > 1000, :]
adata = adata[adata.obs.pct_counts_mt < 13, :]

counts = np.array(adata.X.todense().astype("int32"))

nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
counts = counts[:, nzCounts]

sf = rnaseqFuncs.scranNorm(counts)

countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf, maxThreads=40)
hv = countModel.hv
feat = countModel.residuals[:, hv]
decomp = rnaseqFuncs.permutationPA_PCA(feat, 1, max_rank=100, returnModel=False)

embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0, low_memory=False, metric="correlation").fit_transform(decomp)

import matplotlib.pyplot as plt

plt.figure(dpi=500)
readsPerCell = np.sum(counts, axis=1)
readsPerCell = readsPerCell-np.min(readsPerCell)
plt.scatter(embedding[:, 0], embedding[:, 1], s=1.0, linewidths=0, c=np.sqrt(readsPerCell/readsPerCell.max()))
plt.show()

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
plt.figure(dpi=500)
readsPerCell = countModel.normed[:, adata.var.index[nzCounts] == "PPBP"]
plt.scatter(embedding[:, 0], embedding[:, 1],s=1.0,linewidths=0, c=np.sqrt(readsPerCell/readsPerCell.max()))
plt.show()
# %%
# Run pipeline on Pol II probe expression
# For whatever reason scanpy does not want to load our files
polIICounts = mmread(paths.pbmc10k + "Pol II counts/filtered_feature_bc_matrix/matrix.mtx.gz")
polIICounts = np.array(polIICounts.todense()).astype("int32")
barcodes = pd.read_csv(paths.pbmc10k + "Pol II counts/filtered_feature_bc_matrix/barcodes.tsv.gz", header=None).values
features = pd.read_csv(paths.pbmc10k + "Pol II counts/filtered_feature_bc_matrix/features.tsv.gz", sep="\t", header=None).values
# %%
# Make barcode match between the two analyses
polIICounts = pd.DataFrame(polIICounts.T, index=barcodes.ravel())
foundBarcodes = np.intersect1d(adata.obs.index, polIICounts.index)
polIICounts = polIICounts.loc[foundBarcodes].values
# %%
# Filter not detectable probes
nzCountsPolII = rnaseqFuncs.filterDetectableGenes(polIICounts, readMin=1, expMin=3)
polIICounts = polIICounts[:, nzCountsPolII]
# %%
# Remove
# matchingsf = pd.Series(sf, adata.obs.index).loc[foundBarcodes].values
matchingsf = np.sum(polIICounts, axis=1)
countModelPolII = rnaseqFuncs.RnaSeqModeler().fit(polIICounts, matchingsf, maxThreads=40)
hvPolII = countModelPolII.hv
# %%
from sklearn.decomposition import PCA
feat = countModelPolII.residuals
decomp = PCA(decomp.shape[1]).fit_transform(feat)
# %%
embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0, low_memory=False, metric="correlation").fit_transform(decomp)

import matplotlib.pyplot as plt

plt.figure(dpi=500)
readsPerCell = np.sum(polIICounts, axis=1)
readsPerCell = readsPerCell-np.min(readsPerCell)
plt.scatter(embedding[:, 0], embedding[:, 1],s=1.0,linewidths=0, c=np.sqrt(readsPerCell/readsPerCell.max()))

plt.show()
# %%
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
labels = matrix_utils.graphClustering(decomp, metric="correlation", approx=False, restarts=10, snn=True).astype("str")
colors = sns.color_palette("tab20")
palettePlotly = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
colormap = dict(zip(np.unique(labels), palettePlotly))
df = pd.DataFrame({"x":embedding[:, 0] ,"y":embedding[:, 1], "cluster":labels})
fig = px.scatter(df, x="x", y="y", color="cluster", color_discrete_map=colormap,
                hover_data=['cluster'], width=800, height=800)
fig.update_traces(marker=dict(size=3.0))
fig.write_image(paths.outputDir + "scrnaseq/10k_pbmc_pol2probes_clusterPol2.pdf")
fig.write_html(paths.outputDir + "scrnaseq/10k_pbmc_pol2probes_clusterPol2.pdf" + ".html")
fig.show()
# %%
import pyranges as pr
try:
    os.mkdir(paths.outputDir + "scrnaseq/DE/")
except FileExistsError:
    pass
from lib.utils.reusableUtest import mannWhitneyAsymp
tester = mannWhitneyAsymp(countModelPolII.residuals)
consensuses = pr.read_gtf(paths.tempDir + "Pol2.gtf").as_df()
consensuses.drop(["Source", "Feature", "Frame", "gene_biotype", "transcript_id", "gene_name"], 1, inplace=True)
consensuses.columns = ["Chromosome", "Start", "End","Score","Strand","Name"]
consensuses = consensuses[["Chromosome", "Start", "End", "Name", "Strand"]]
consensuses.index = consensuses["Name"].astype("int")
consensuses = consensuses.loc[features[:, 1]]
# %%
pctThreshold = 0.1
lfcMin = 0.25
matchinglabels = pd.Series(labelsGenes, adata.obs.index).loc[foundBarcodes].values
for i in np.unique(matchinglabels):
    print(i)
    grp = (matchinglabels == i).astype(int)
    res2 = tester.test(grp, "less")
    sig = fdrcorrection(res2[1])[0]
    minpct = np.mean(polIICounts[matchinglabels == i] > 0.5, axis=0) > max(0.1, 1.5/grp.sum())
    fc = np.mean(countModelPolII.normed[matchinglabels == i], axis=0) / (1e-9+np.mean(countModelPolII.normed[matchinglabels != i], axis=0))
    lfc = np.log2(fc) > lfcMin
    sig = sig & lfc & minpct
    print(sig.sum())
    res = pd.DataFrame(res2[::-1], columns=consensuses.index[nzCountsPolII], index=["pval", "stat"]).T
    res["Upreg"] = sig.astype(int)
    res.to_csv(paths.outputDir + f"scrnaseq/DE/res_{i}.csv")
    test = consensuses[nzCountsPolII][sig]
    test.to_csv(paths.outputDir + f"scrnaseq/DE/bed_{i}.bed", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
# %%
tester = mannWhitneyAsymp(countModel.residuals)
# %%
pctThreshold = 0.1
lfcMin = 0.25
matchinglabels = labelsGenes
for i in np.unique(matchinglabels):
    print(i)
    grp = (matchinglabels == i).astype(int)
    res2 = tester.test(grp, "less")
    sig = fdrcorrection(res2[1])[0]
    minpct = np.mean(counts[matchinglabels == i] > 0.5, axis=0) > max(0.1, 1.5/grp.sum())
    fc = np.mean(countModel.normed[matchinglabels == i], axis=0) / (1e-9+np.mean(countModel.normed[matchinglabels != i], axis=0))
    lfc = np.log2(fc) > lfcMin
    sig = sig & lfc & minpct
    print(sig.sum())
    res = pd.DataFrame(res2[::-1], columns=adata.var.index[nzCounts], index=["pval", "stat"]).T
    res["Upreg"] = sig.astype(int)
    order = np.argsort(res["pval"])
    res = res.iloc[np.argsort(res["pval"])]
    res.to_csv(paths.outputDir + f"scrnaseq/DE/res_genes_{i}.csv")
    test = pd.Series(adata.var.index[nzCounts][order][sig[order]])
    test.to_csv(paths.outputDir + f"scrnaseq/DE/genes_{i}.csv", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
# %%
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

print(adjusted_rand_score(matchinglabels, labels), adjusted_mutual_info_score(matchinglabels, labels))
# %%
