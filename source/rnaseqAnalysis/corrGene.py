# %%
import os

import fastcluster
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import sklearn.metrics as metrics
import umap
from lib import rnaseqFuncs
from lib.utils import matrix_utils, plot_utils
from matplotlib.patches import Patch
from scipy.spatial.distance import dice
from scipy.stats import chi2, rankdata
from settings import params, paths
from statsmodels.stats.multitest import fdrcorrection

folder = paths.outputDir + "dist_to_genes/corr_expr/"
try:
    os.mkdir(paths.outputDir + "dist_to_genes/corr_expr/")
except:
    pass
# %%
allAnnots = pd.read_csv(paths.tcgaAnnot, 
                        sep="\t", index_col=0)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
try:
    os.mkdir(paths.outputDir + "rnaseq/TumorVsNormal/")
except FileExistsError:
    pass
perCancerDE = pd.DataFrame()
wAccs = pd.DataFrame()
recalls = pd.DataFrame()
precisions = pd.DataFrame()
studiedConsensusesCase = dict()
cases = allAnnots["project_id"].unique()
# %%
# Compute DE, UMAP, and predictive model CV per cancer
case = "TCGA-KICH"
print(case)
# Select only relevant files and annotations
annotation = pd.read_csv(paths.tcgaAnnot, 
                        sep="\t", index_col=0)
annotation = annotation[annotation["project_id"] == case]
annotation.drop_duplicates("Sample ID", inplace=True)
dlFiles = os.listdir(paths.countsTCGA + "500centroid/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
ids = np.array([f.split(".")[0] for f in dlFiles])
inAnnot = np.isin(ids, annotation.index)
ids = ids[inAnnot]
dlFiles = np.array(dlFiles)[inAnnot]
annotation = annotation.loc[ids]
labels = []
for a in annotation["Sample Type"]:
    if a == "Solid Tissue Normal":
        labels.append(0)
    else:
        labels.append(1)
labels = np.array(labels)
if len(labels) < 10 or np.any(np.bincount(labels) < 10):
    print(case, "not enough samples")
# Read files and setup data matrix
counts = []
allReads = []
order = []
for f in dlFiles:
    try:
        fid = f.split(".")[0]
        status = pd.read_csv(paths.countsTCGA + "500centroid/" + fid + ".counts.summary",
                            header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(paths.countsTCGA + "500centroid/" + f, header=None, skiprows=2).values.astype("int32"))
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(fid)
    except:
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
# %%
# Read table and annotation
geneTable = pd.read_hdf(paths.tcgaGeneCounts)
geneTableAnnot = pd.read_csv(paths.tcgaAnnotCounts, index_col="Sample ID", sep="\t")
geneTableAnnot = geneTableAnnot[~geneTableAnnot.index.duplicated(keep='first')]
used = geneTableAnnot.loc[annotation["Sample ID"]]["File ID"]
used = used[~used.index.duplicated(keep='first')]
usedTable = geneTable[used].astype("int32").iloc[:-5].T
nzCounts = rnaseqFuncs.filterDetectableGenes(usedTable.values, readMin=1, expMin=3)
usedTable = usedTable.loc[:, nzCounts]
# %%
# Get size factors
sf = rnaseqFuncs.deseqNorm(usedTable.values)
sf /= sf.mean()
# %%
# Compute NB model and residuals
countModel = rnaseqFuncs.RnaSeqModeler().fit(usedTable.values, sf, maxThreads=40)
hv = countModel.hv

# %%
# PCA on residuals
feat = countModel.residuals[:, hv]
decomp = rnaseqFuncs.permutationPA_PCA(feat)
labels = geneTableAnnot.loc[used.index]["Sample Type"] == "Solid Tissue Normal"
labels = np.array(labels).astype(int)
matrix_utils.looKnnCV(decomp, labels, "correlation", 1)
# %%
# Plot UMAP
embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(decomp)
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels)
# %%
# Find matching 
hasAnnot = np.isin(used.index.values, annotation["Sample ID"].values)
# %%
# Detectable Pol II probes in dataset
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=3)
allCounts = allCounts[:, nzCounts]
# %%
# Compute NB model and residuals
countModel2 = rnaseqFuncs.RnaSeqModeler().fit(allCounts, sf, maxThreads=40)
hv2 = countModel2.hv

# %%
# PCA on residuals
feat = countModel2.residuals[:, hv2]
decomp2 = rnaseqFuncs.permutationPA_PCA(feat)
labels = geneTableAnnot.loc[used.index]["Sample Type"] == "Solid Tissue Normal"
labels = np.array(labels).astype(int)
matrix_utils.looKnnCV(decomp2, labels, "correlation", 1)
# Plot UMAP
embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(decomp2)
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels)
# %%
import plotly.express as px
np.random.seed(0)
def associationComputations(countTable, geneCounts, sf, assocTable, ensemblToID, pearsonG=None, pearsonP=None):
    # Find ensembl ID
    geneStableID = [id.split(".")[0] for id in usedTable.columns]
    usedTable.columns = geneStableID
    valid = np.isin(assocTable["gene_name"].values, ensemblToID.index)
    assocTable_cv = assocTable[valid]
    assocTable_cv["ensID"] = ensemblToID.loc[assocTable_cv["gene_name"].values].values
    valid = np.isin(assocTable_cv["ensID"].values, geneStableID)
    assocTable_cv = assocTable_cv[valid]
    # Find associated gene, re-associate due to removed probes, compute normalized counts
    nzRemap = pd.Series(data=np.arange(nzCounts.sum()), index=np.arange(len(consensuses))[nzCounts])
    inMap = np.isin(assocTable_cv["Name"].values, np.arange(len(consensuses))[nzCounts])
    selected = nzRemap.loc[assocTable_cv["Name"].values[inMap]]
    x = (countTable/sf.reshape(-1,1))[:, selected]
    resScaled = geneCounts/sf.reshape(-1,1)
    tabResiduals = pd.DataFrame(resScaled, columns=geneStableID)
    y = tabResiduals.loc[:, assocTable_cv["ensID"].values[inMap]].values
    # Distribution of correlations between tail of gene Pol II and associated gene 
    from scipy.spatial.distance import correlation
    from scipy.stats import rankdata, spearmanr, pearsonr
    xrank = rankdata(x, axis=0)

    yrank = rankdata(y, axis=0)
    correlations = np.array([pearsonr(xrank[:, i],yrank[:, i]) for i in range(y.shape[1])])
    fdrCorr = fdrcorrection(correlations[:, 1])[0]
    firstSig = np.argmax(fdrCorr)
    threshold = correlations[firstSig][0]
    if fdrCorr.sum() == 0:
        n = len(yrank)
        dist = beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
        threshold = dist.ppf(0.025/n)
    plt.figure(dpi=500)
    plt.hist(correlations[:, 0], 50, density=True)
    plt.vlines([threshold, -threshold], plt.ylim()[0], plt.ylim()[1], colors=[1,0,0], linestyles="dashed")
    plt.xlabel("Distribution of per Pol II probe / tailed gene\nspearman correlation of expression.")
    plt.title("Correlation between gene and tail of gene Pol II probes (5kb)")
    plt.savefig(folder + "corr_tail.pdf")
    print(pearsonr(xrank.ravel(), yrank.ravel()))
    plt.show()
    plt.close()
    print(x.shape)
    print(y.shape)
    print(correlations.shape)
    # Corr expr / correlation
    fig = px.scatter(x=np.log(np.mean(y, axis=0)), y=correlations[:, 0],
                title=str(spearmanr(np.log(np.mean(y, axis=0)), correlations[:, 0])),
                labels=dict(x="Gene log(mean norm expression)", y="Spearman r"),
                marginal_x="histogram", trendline="lowess", trendline_color_override="red")
    fig.show()
    fig.write_image(folder+"expr_vs_corr.pdf")
    fig.write_html(folder + "expr_vs_corr.pdf" + ".html")
    # Distribution of correlations between tail of gene Pol II and RANDOMIZED gene 
    from scipy.spatial.distance import correlation
    from scipy.stats import rankdata, beta, pearsonr
    permutedY = np.random.permutation(y)
    xrank = rankdata(x, axis=0)
    yrank = rankdata(permutedY, axis=0)
    correlations = np.array([pearsonr(xrank[:, i],yrank[:, i]) for i in range(y.shape[1])])
    fdrCorr = fdrcorrection(correlations[:, 1])[0]
    firstSig = np.argmax(fdrCorr)
    threshold = correlations[firstSig, 0]
    if fdrCorr.sum() == 0:
        n = len(yrank)
        dist = beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
        threshold = dist.ppf(0.025/n)
    print(pearsonr(xrank.ravel(), yrank.ravel()))
    plt.figure(dpi=500)
    plt.hist(correlations[:, 0], 50, density=True)
    plt.vlines([threshold, -threshold], plt.ylim()[0], plt.ylim()[1], colors=[1,0,0], linestyles="dashed")
    plt.xlabel("Distribution of per Pol II probe / random gene\nspearman correlation of expression.")
    plt.title("Correlation between gene and tail of gene Pol II probes (5kb)")
    plt.savefig(folder + "corr_tail_permute.pdf")
    fig = px.scatter(x=np.log(np.mean(permutedY, axis=0)), y=correlations[:, 0],
            title=str(spearmanr(np.log(np.mean(y, axis=0)), correlations[:, 0])),
            labels=dict(x="Gene log(mean norm expression)", y="Spearman r"),
            marginal_x="histogram", trendline="lowess", trendline_color_override="red")
    fig.show()
    fig.write_image(folder+"expr_vs_corr_permute.pdf")
    fig.write_html(folder + "expr_vs_corr_permute.pdf" + ".html")

# Load tail of gene pol 2
polII_tail = pd.read_csv(paths.outputDir + "/dist_to_genes/pol2_5000_TSS_ext.bed", sep="\t")
ensemblToID = pd.read_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data/ensembl_toGeneId.tsv", sep="\t", index_col="Gene name")
ensemblToID = ensemblToID[~ensemblToID.index.duplicated(keep='first')]
associationComputations(allCounts, usedTable.values, sf, polII_tail, ensemblToID)
# %%
