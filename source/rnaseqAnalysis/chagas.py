# %%
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
from scipy.spatial.distance import dice
import matplotlib as mple
import fastcluster
import sklearn.metrics as metrics
import scipy.stats as ss

countDir = paths.outputDir + "rnaseq/chagas_counts/"
try:
    os.mkdir(paths.outputDir + "rnaseq/")
except FileExistsError:
    pass
try:
    os.mkdir(paths.outputDir + "rnaseq/chagas_rnaseq/")
except FileExistsError:
    pass
consensusesHg19 = pd.read_csv(paths.outputDir + "consensusesHg19.bed", sep="\t", header=None, index_col=3)[[0,1,2]]
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None, index_col=3)[[0,1,2]]
consensuses.columns = ["Chromosome", "Start", "End"]
# %%
from lib.pyGREATglm import pyGREAT as pyGREAT
enricher = pyGREAT(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
# %%
dlFiles = os.listdir(countDir + "500centroid/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
allStatus = []
for f in dlFiles:
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
counts = np.concatenate(counts, axis=1).T
# %%
labels = []
for sample in order:
    if sample.startswith("DCM"):
        labels.append("DCM")
    elif sample.startswith("CTRL"):
        labels.append("CTRL")
    else:
        labels.append("sevCCC")
labels = np.array(labels)
# %%
nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
counts = counts[:, nzCounts]

# %%
sf = rnaseqFuncs.scranNorm(counts)
# %%
from trash.countModeler import RnaSeqModeler

countModel = RnaSeqModeler().fit(counts, sf)
from joblib.externals.loky import get_reusable_executor
get_reusable_executor().shutdown(wait=True)
hv = countModel.hv
# %%
import plotly.express as px
feat = countModel.residuals[:, hv]
decomp = rnaseqFuncs.permutationPA_PCA(feat)
print(matrix_utils.looKnnCV(feat, pd.factorize(labels)[0], "correlation", 1))
df = pd.DataFrame(decomp[:,:2], columns=["x","y"])
df["Cond"] = labels
df = df.sample(frac=1)
fig = px.scatter(df, x="x", y="y", color="Cond", hover_data=['Cond'], 
                width=1200, height=800)
fig.update_traces(marker=dict(size=0.5/np.sqrt(len(df)/7500)))
fig.show()
fig.write_image(paths.outputDir + "rnaseq/chagas_rnaseq/PCA_samples.pdf")
fig.write_html(paths.outputDir + "rnaseq/chagas_rnaseq/PCA_samples.pdf" + ".html")
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=int(np.sqrt(len(counts)))+1, random_state=42, low_memory=False, metric="correlation").fit_transform(feat)
df = pd.DataFrame(embedding, columns=["x","y"])
df["Cond"] = labels
df = df.sample(frac=1)
fig = px.scatter(df, x="x", y="y", color="Cond", hover_data=['Cond'], 
                width=1200, height=800)
fig.update_traces(marker=dict(size=0.5/np.sqrt(len(df)/7500)))
fig.show()
fig.write_image(paths.outputDir + "rnaseq/chagas_rnaseq/UMAP_samples.pdf")
fig.write_html(paths.outputDir + "rnaseq/chagas_rnaseq/UMAP_samples.pdf" + ".html")
# %%
# Control vs Dilated CardioMyopathy
selected = (labels == "CTRL") | (labels == "DCM")
DEtab = rnaseqFuncs.deseqDE(counts[selected], sf[selected], labels=labels[selected], colNames=np.arange(selected.sum()))
minpctM = np.mean(counts[labels == "CTRL"] > 0.5, axis=0) > max(0.1, 1.99/(labels == "CTRL").sum())
minpctP = np.mean(counts[labels == "DCM"] > 0.5, axis=0) > max(0.1, 1.99/ (labels == "DCM").sum())
minpct = minpctM | minpctP
DEDCM = (DEtab["padj"] < 0.05) & (np.abs(DEtab["log2FoldChange"]) > 0.25) & minpct
DEconsHg19 = consensusesHg19[nzCounts].iloc[DEtab[DEDCM].sort_values("padj").index]
DEconsHg19["Id"] = DEconsHg19.index
DEconsHg19["LogQ"] = DEtab[DEDCM].sort_values("padj")["padj"].values
DEconsHg19.to_csv(paths.outputDir + "rnaseq/chagas_rnaseq/DCM_DE_hg19.bed", sep="\t", header=None, index=None)
DEcons = consensuses.loc[DEconsHg19.index]
DEcons["Id"] = DEconsHg19.index
DEcons["LogQ"] = DEtab[DEDCM].sort_values("padj")["padj"].values
DEcons.to_csv(paths.outputDir + "rnaseq/chagas_rnaseq/DCM_DE.bed", sep="\t", header=None, index=None)

pvals = enricher.findEnriched(DEcons, background=consensuses)
enricher.plotEnrichs(pvals, savePath=paths.outputDir + "rnaseq/chagas_rnaseq/DCM_DE_great_bar.pdf")
enricher.clusterTreemap(pvals, score="-log10(pval)", metric="yule", output=paths.outputDir + "rnaseq/chagas_rnaseq/DCM_DE_great_treemap.pdf")
# %%
# DE Control vs severe
selected = (labels == "CTRL") | (labels == "sevCCC")
DEtab = rnaseqFuncs.deseqDE(counts[selected], sf[selected], labels=labels[selected], colNames=np.arange(selected.sum()))
minpctM = np.mean(counts[labels == "CTRL"] > 0.5, axis=0) > max(0.1, 1.99/(labels == "CTRL").sum())
minpctP = np.mean(counts[labels == "sevCCC"] > 0.5, axis=0) > max(0.1, 1.99/ (labels == "sevCCC").sum())
minpct = minpctM | minpctP
DECCC = (DEtab["padj"] < 0.05) & (np.abs(DEtab["log2FoldChange"]) > 0.25) & minpct
DEconsHg19 = consensusesHg19[nzCounts].iloc[DEtab[DECCC].sort_values("padj").index]
DEconsHg19["Id"] = DEconsHg19.index
DEconsHg19["LogQ"] = DEtab[DECCC].sort_values("padj")["padj"].values
DEconsHg19.to_csv(paths.outputDir + "rnaseq/chagas_rnaseq/sevCCC_DE_hg19.bed", sep="\t", header=None, index=None)
DEcons = consensuses.loc[DEconsHg19.index]
DEcons["Id"] = DEconsHg19.index
DEcons["LogQ"] = DEtab[DECCC].sort_values("padj")["padj"].values
DEcons.to_csv(paths.outputDir + "rnaseq/chagas_rnaseq/sevCCC_DE.bed", sep="\t", header=None, index=None)

pvals = enricher.findEnriched(DEcons, background=consensuses)
enricher.plotEnrichs(pvals, savePath=paths.outputDir + "rnaseq/chagas_rnaseq/sevCCC_DE_great_bar.pdf")
enricher.clusterTreemap(pvals, score="-log10(pval)", metric="yule", output=paths.outputDir + "rnaseq/chagas_rnaseq/sevCCC_DE_great_treemap.pdf")
# %% 
# Only sevCCC no DCM
ccc_nocdm = DECCC & np.logical_not(DEDCM)
DEconsHg19 = consensusesHg19[nzCounts].iloc[DEtab[ccc_nocdm].sort_values("padj").index]
DEconsHg19["Id"] = DEconsHg19.index
DEconsHg19["LogQ"] = DEtab[ccc_nocdm].sort_values("padj")["padj"].values
DEconsHg19.to_csv(paths.outputDir + "rnaseq/chagas_rnaseq/CCC_noDCM_DE_hg19.bed", sep="\t", header=None, index=None)
DEcons = consensuses.loc[DEconsHg19.index]
DEcons["Id"] = DEconsHg19.index
DEcons["LogQ"] = DEtab[ccc_nocdm].sort_values("padj")["padj"].values
DEcons.to_csv(paths.outputDir + "rnaseq/chagas_rnaseq/CCC_noDCM_DE.bed", sep="\t", header=None, index=None)

pvals = enricher.findEnriched(DEcons, background=consensuses)
enricher.plotEnrichs(pvals, savePath=paths.outputDir + "rnaseq/chagas_rnaseq/sevCCC_noDCM_DE_great_bar.pdf")
enricher.clusterTreemap(pvals, score="-log10(pval)", metric="yule", output=paths.outputDir + "rnaseq/chagas_rnaseq/sevCCC_noDCM_DE_great_treemap.pdf")
# %%
# Only sevCCC & DCM
ccc_cdm = DECCC & DEDCM
DEconsHg19 = consensusesHg19[nzCounts].iloc[DEtab[ccc_cdm].sort_values("padj").index]
DEconsHg19["Id"] = DEconsHg19.index
DEconsHg19["LogQ"] = DEtab[ccc_cdm].sort_values("padj")["padj"].values
DEconsHg19.to_csv(paths.outputDir + "rnaseq/chagas_rnaseq/CCC_DCM_DE_hg19.bed", sep="\t", header=None, index=None)
DEcons = consensuses.loc[DEconsHg19.index]
DEcons["Id"] = DEconsHg19.index
DEcons["LogQ"] = DEtab[ccc_cdm].sort_values("padj")["padj"].values
DEcons.to_csv(paths.outputDir + "rnaseq/chagas_rnaseq/CCC_DCM_DE.bed", sep="\t", header=None, index=None)

pvals = enricher.findEnriched(DEcons, background=consensuses)
enricher.plotEnrichs(pvals, savePath=paths.outputDir + "rnaseq/chagas_rnaseq/sevCCC_DCM_DE_great_bar.pdf")
enricher.clusterTreemap(pvals, score="-log10(pval)", metric="yule", output=paths.outputDir + "rnaseq/chagas_rnaseq/sevCCC_DCM_DE_great_treemap.pdf")
# %%
# Heatmap
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(feat, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(feat.T, "correlation")
logCounts = np.log10(1+countModel.normed)
plot_utils.plotHC(logCounts.T[hv], labels, (countModel.normed).T[hv],  
                  rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/chagas_rnaseq/chagas_HM_hvg.pdf")
plt.show()
# %%
# Heatmap all DE
allDE = DEDCM | DECCC
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(countModel.residuals[:, allDE], "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.residuals[:, allDE].T, "correlation")
logCounts = np.log10(1+countModel.normed)
plot_utils.plotHC(logCounts.T[allDE], labels, (countModel.normed).T[allDE],  
                  rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/chagas_rnaseq/chagas_HM_DE.pdf")
plt.show()
# %%
# Heatmap DE CCC not DE DCM
allDE = ccc_nocdm
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(countModel.residuals[:, allDE], "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.residuals[:, allDE].T, "correlation")
logCounts = np.log10(1+countModel.normed)
plot_utils.plotHC(logCounts.T[allDE], labels, (countModel.normed).T[allDE],  
                  rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/chagas_rnaseq/chagas_HM_DE_CCC_noCDM.pdf")
plt.show()
# %%
