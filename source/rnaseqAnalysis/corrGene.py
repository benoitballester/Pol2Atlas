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

# %%
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
case = "TCGA-ESCA"
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
nzCounts = rnaseqFuncs.filterDetectableGenes(usedTable.values, readMin=1, expMin=2)
usedTable = usedTable.loc[:, nzCounts]
# %%
# Get size factors
sf = rnaseqFuncs.deseqNorm(usedTable.values)
sf /= sf.mean()
# %%
# Compute NB model and residuals
countModel = rnaseqFuncs.RnaSeqModeler().fit(usedTable.values, sf)
hv = countModel.hv

# %%
# PCA on residuals
feat = countModel.residuals[:, hv]
decomp, model = rnaseqFuncs.permutationPA_PCA(feat, returnModel=True)
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
# Compute NB model and residuals for Pol II probes
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=2)
allCounts = allCounts[:, nzCounts]
countModelPol2 = rnaseqFuncs.RnaSeqModeler().fit(allCounts, sf[hasAnnot])
# %%
polII_tail = pd.read_csv("/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/dist_to_genes/pol2_5000_TES_ext.bed", sep="\t")
ensemblToID = pd.read_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data/ensembl_toGeneId.tsv", sep="\t", index_col="Gene name")
ensemblToID = ensemblToID[~ensemblToID.index.duplicated(keep='first')]
# %%
geneStableID = [id.split(".")[0] for id in usedTable.columns]
usedTable.columns = geneStableID
valid = np.isin(polII_tail["gene_name"].values, ensemblToID.index)
polII_tail_cv = polII_tail[valid]
polII_tail_cv["ensID"] = ensemblToID.loc[polII_tail_cv["gene_name"].values].values
valid = np.isin(polII_tail_cv["ensID"].values, geneStableID)
polII_tail_cv = polII_tail_cv[valid]
# %%
x = countModelPol2.residuals[:, polII_tail_cv["Name"].values]
tabResiduals = pd.DataFrame(countModel.residuals, columns=geneStableID)
y = tabResiduals.loc[:, polII_tail_cv["ensID"].values].values
# %%
from scipy.spatial.distance import correlation
from scipy.stats import rankdata, spearmanr, pearsonr
xrank = rankdata(x, axis=0)
yrank = rankdata(y, axis=0)
correlations = 1-np.array([correlation(xrank[:, i],yrank[:, i]) for i in range(y.shape[1])])
plt.figure(dpi=500)
plt.hist(correlations, 20)
plt.xlabel("Distribution of per Pol II probe spearman correlation")
plt.title("Correlation between gene and tail of gene Pol II probes (5kb)")
# %%
pearsonr(xrank.ravel(), yrank.ravel())
# %%
np.random.seed(42)
subset = np.random.choice(len(x.ravel()), 200000)
plt.figure(dpi=500)
plt.scatter(x.ravel()[subset], y.ravel()[subset],s=0.1,linewidth=0.0)
plt.xlabel("Pol II probe pearson residuals")
plt.ylabel("Nearest gene pearson residuals")

# %%
