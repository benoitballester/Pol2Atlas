# %%
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
from scipy.spatial.distance import dice
import matplotlib as mpl
import fastcluster
import sklearn.metrics as metrics
import scipy.stats as ss

countDir = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/rnaseq/gtex_counts/"
try:
    os.mkdir(paths.outputDir + "rnaseq/gtex_rnaseq/")
except FileExistsError:
    pass
# %%
annotation = pd.read_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/GTex/tsvs/sample.tsv", 
                        sep="\t", index_col="specimen_id")
colors = pd.read_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/GTex/colors.txt", 
                        sep="\t", index_col="tissue_site_detail")
dlFiles = os.listdir(countDir + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
allStatus = []
for f in dlFiles:
    if not annotation.loc[f.split(".")[0]]["tissue_type"] == "Kidney":
        continue
    try:
        id = ".".join(f.split(".")[:-3])
        # countsBG.append(pd.read_csv(paths.countDirectory + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                                header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values.astype("int32"))
        allStatus.append(status)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(f.split(".")[0])
    except:
        print(f, "missing")
        continue
allReads = np.array(allReads)
countsGtexKidney = np.concatenate(counts, axis=1).T
ann, eq = pd.factorize(annotation.loc[order]["tissue_type"])
# %%
# Select only relevant files and annotations
annotationTCGA = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
kidneys = ["TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP"]
isproj = [annotationTCGA["project_id"].values == c for c in kidneys]
isKidney = np.any(isproj, axis=0)
annotationTCGA = annotationTCGA[isKidney]
dlFiles = os.listdir(paths.countsTCGA + "500centroid/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
ids = np.array([f.split(".")[0] for f in dlFiles])
inAnnot = np.isin(ids, annotationTCGA.index)
ids = ids[inAnnot]
dlFiles = np.array(dlFiles)[inAnnot]
annotationTCGA = annotationTCGA.loc[ids]
labels = []
for a in annotationTCGA["Sample Type"]:
    if a == "Solid Tissue Normal":
        labels.append(0)
    else:
        labels.append(1)
labels = np.array(labels)
annotationTCGA = annotationTCGA[labels==0]
dlFiles = dlFiles[labels==0]
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
labels = []
annotationTCGA = annotationTCGA.loc[order]
for a in annotationTCGA["Sample Type"]:
    if a == "Solid Tissue Normal":
        labels.append(0)
    else:
        labels.append(1)
labels = np.array(labels)
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
# %%
merged = np.concatenate([allCounts,countsGtexKidney], axis=0)
# %%
numExp = np.array(isproj).sum(axis=1)
labels = np.concatenate([labels, [0]*89])
datasetOrig = list(annotationTCGA["project_id"].loc[order]) + ["GTEx normal"]*89
orig = ["TCGA"]*len(annotationTCGA["project_id"].loc[order]) + ["GTEx normal"]*89
nzCounts = rnaseqFuncs.filterDetectableGenes(merged, readMin=1, expMin=3)
counts = merged[:, nzCounts]
print(counts.shape)
# %%
sf = rnaseqFuncs.scranNorm(counts)
# %%
# Not Correcting for dataset
countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf, maxThreads=50)
hv = countModel.hv
feat = countModel.residuals[:, hv]
decomp = rnaseqFuncs.permutationPA_PCA(feat, 1, whiten=True, returnModel=False)
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(decomp)

import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
df["Dataset"] = datasetOrig
df["State"] = labels
fig = px.scatter(df, x="x", y="y", color="Dataset", 
                hover_data=['State'], symbol="Dataset", width=1200, height=800)
fig.update_traces(marker=dict(size=10))
fig.show()
# %%
# Correcting for dataset bias
design = pd.get_dummies(orig)
countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf, design, maxThreads=50)
hv = countModel.hv
feat = countModel.residuals[:, hv]
decomp = rnaseqFuncs.permutationPA_PCA(feat, 1, whiten=True, returnModel=False)
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(decomp)
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
df["Dataset"] = datasetOrig
df["State"] = labels.astype("str")
fig = px.scatter(df, x="x", y="y", color="Dataset", 
                hover_data=['State'], symbol="Dataset", width=1200, height=800)
fig.update_traces(marker=dict(size=10))
fig.show()
# %%
