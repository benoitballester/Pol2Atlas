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
import matplotlib as mpl
import fastcluster
import sklearn.metrics as metrics
import scipy.stats as ss
import plotly.express as px
import plotly.graph_objects as go

countDir = paths.countsGTEx
try:
    os.mkdir(paths.outputDir + "rnaseq/")
except FileExistsError:
    pass
try:
    os.mkdir(paths.outputDir + "rnaseq/gtex_rnaseq_heart_DE/")
except FileExistsError:
    pass
# %%
# Read count files
annotation = pd.read_csv(paths.gtexData + "tsvs/sample_annot.tsv", 
                        sep="\t", index_col="specimen_id")
colors = pd.read_csv(paths.gtexData + "colors.txt", 
                        sep="\t", index_col="tissue_site_detail")
annotation = annotation[annotation.loc[:, "tissue_type"] == "Heart"]
dlFiles = os.listdir(countDir + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
allStatus = []
dlFiles = [i.split(".")[0] for i in dlFiles]
dlFiles = np.intersect1d(dlFiles, annotation.index.values)
for f in dlFiles:
    try:
        f += ".Aligned.sortedByCoord.out.patched.md.bam.counts.txt.gz"
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
conv = pd.read_csv(paths.tissueToSimplified, sep="\t", index_col="Tissue")
annTxt = annotation.loc[order]["tissue_type_detail"].values
ann, eq = pd.factorize(annTxt)
# %%
np.random.seed(42)
fullAnn = annotation.loc[order]["tissue_type_detail"].values
annFull, eqFull = pd.factorize(fullAnn)
sampleSizes = np.bincount(annFull)
subsample1 = np.random.choice(sampleSizes[1], 3)
subsample2 = np.random.choice(sampleSizes[0], 3)
subTable = np.concatenate([counts[annFull == 0][subsample1], counts[annFull == 1][subsample2]], axis=0)
nzCountsSub = rnaseqFuncs.filterDetectableGenes(subTable, readMin=1, expMin=3)
subTable = subTable[:, nzCountsSub]
sf = rnaseqFuncs.scranNorm(subTable)
countModel = rnaseqFuncs.RnaSeqModeler().fit(subTable, sf)
hv = countModel.hv
feat = countModel.residuals
decomp = rnaseqFuncs.permutationPA_PCA(feat, 10, returnModel=False, mincomp=2)
plt.scatter(decomp[:,0],decomp[:,1], c=[0]*3+[1]*3)
matrix_utils.looKnnCV(decomp, np.array([0]*3+[1]*3), "euclidean", 1)
# %%
DEtab = rnaseqFuncs.deseqDE(subTable, sf, labels=np.array([eq[0]]*3 + [eq[1]]*3), 
                            colNames=np.arange(subTable.shape[0]))
DE = (DEtab["padj"] < 0.05) & (np.abs(DEtab["log2FoldChange"]) > 0.25)
# %%
rowOrder, rowLink = matrix_utils.twoStagesHClinkage(decomp)
colOrder, colLink = matrix_utils.twoStagesHClinkage(feat.T[DE.values])
logCounts = countModel.residuals
plot_utils.plotHC(logCounts.T[DE.values], np.array([eq[0]]*3 + [eq[1]]*3), (countModel.normed).T[DE.values], rescale="3SD",  
                  rowOrder=rowOrder, colOrder=colOrder, cmap="vlag")
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq_heart_DE/n3_heatmap.pdf")
plt.show()
# %%
# Remove unexpressed probes, normalize, compute pearson residuals
nzCounts0 = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
counts = counts[:, nzCounts0]
sf = rnaseqFuncs.scranNorm(counts)
countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf)
hv = countModel.hv
feat = countModel.residuals[:, hv]
decomp = rnaseqFuncs.permutationPA_PCA(feat, 10, returnModel=False)
# KNN classifier
acc = matrix_utils.looKnnCV(feat, annFull, "correlation", 1)
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(decomp)
df = pd.DataFrame(embedding, columns=["x","y"])
df["Organ"] = annotation.loc[order]["tissue_type"].values
df["Organ detailled"] = annotation.loc[order]["tissue_type_detail"].values
df = df.sample(frac=1)
colormap = dict(zip(colors.index, colors["color_hex"])) 
batchSize = 500
batches = []
for i in range(int(len(df)/batchSize+2)):
    batches.append(px.scatter(df[i*batchSize:(i+1)*batchSize], x="x", y="y", color="Organ detailled", 
                              color_discrete_map=colormap))
allData = batches[0].data
for i in batches[1:]:
    allData = allData + i.data
fig = go.Figure(data = allData, layout={"title": f"54 Tissue KNN classification balanced accuracy : {acc}",
                                        "width":1200, "height":800})
fig.update_traces(marker=dict(size=9/(len(df)/750)))
fig.show()
# %%
# Performs random sampling at different sample sizes, then perform DE
sampleSizes = np.bincount(annFull)
tab1 = counts[annFull == 0]
sf1 = sf[annFull == 0]
tab2 = counts[annFull == 1]
sf2 = sf[annFull == 1]
reps = 10
results = np.zeros((counts.shape[1], reps), dtype=bool)
np.random.seed(42)
rsize = 3
for i in range(reps):
    subsample1 = np.random.choice(sampleSizes[0], rsize)
    subsample2 = np.random.choice(sampleSizes[1], rsize)
    labels = np.array([eq[0]]*rsize + [eq[1]]*rsize)
    countTable = np.concatenate([tab1[subsample1], tab2[subsample2]], axis=0)
    nzCounts = rnaseqFuncs.filterDetectableGenes(countTable, readMin=1, expMin=3)
    countTable = countTable[:, nzCounts]
    sfTable = np.concatenate([sf1[subsample1], sf2[subsample2]], axis=0)
    DEtab = rnaseqFuncs.deseqDE(countTable, sfTable, labels=labels, 
                                colNames=np.arange(countTable.shape[0]))
    minpctM = np.mean(countTable[labels == eq[0]] > 0.5, axis=0) > max(0.1, 1.99/(labels == eq[0]).sum())
    minpctP = np.mean(countTable[labels == eq[1]] > 0.5, axis=0) > max(0.1, 1.99/ (labels == eq[1]).sum())
    minpct = minpctM | minpctP
    DE = (DEtab["padj"] < 0.05) & (np.abs(DEtab["log2FoldChange"]) > 0.25) & minpct
    results[nzCounts, i] = DE

# %%
means = np.mean(results, axis=1)
sds = np.std(results, axis=0)
# %%
# GO enrichments
from lib.pyGREATglm import pyGREAT as pyGREAT
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None, index_col=3)[[0,1,2]]
consensuses.columns = ["Chromosome", "Start", "End"]
enricher = pyGREAT(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
pvals = enricher.findEnriched(consensuses[nzCounts0][(means >= 0.5)], background=consensuses)
enricher.plotEnrichs(pvals)
DElist = consensuses[nzCounts0][(means >= 0.5)]
DElist.to_csv(paths.outputDir + "rnaseq/gtex_rnaseq_heart_DE/n3_DE_heart.bed")
enricher.clusterTreemap(pvals, score="-log10(pval)", metric="yule", output=paths.outputDir + "rnaseq/gtex_rnaseq_heart_DE/DE_treemap.pdf")
# %%
# Performs random sampling at different sample sizes, then perform DE
sampleSizes = np.bincount(annFull)
tab1 = counts[annFull == 0]
sf1 = sf[annFull == 0]
tab2 = counts[annFull == 1]
sf2 = sf[annFull == 1]
res1 = countModel.residuals[annFull == 0]
res2 = countModel.residuals[annFull == 1]
reps = 10
nPts = 15
allPts = np.sort(np.unique(np.geomspace(3, 100, nPts).astype(int)))
results = np.zeros((counts.shape[1], nPts, reps), dtype=bool)
resultsTtest = np.zeros((counts.shape[1], nPts, reps), dtype=bool)
np.random.seed(42)
for i in range(reps):
    for j, rsize in enumerate(allPts):
        subsample1 = np.random.choice(sampleSizes[0], rsize)
        subsample2 = np.random.choice(sampleSizes[1], rsize)
        labels = np.array([eq[0]]*rsize + [eq[1]]*rsize)
        countTable = np.concatenate([tab1[subsample1], tab2[subsample2]], axis=0)
        nzCounts = rnaseqFuncs.filterDetectableGenes(countTable, readMin=1, expMin=3)
        countTable = countTable[:, nzCounts]
        sfTable = np.concatenate([sf1[subsample1], sf2[subsample2]], axis=0)
        DEtab = rnaseqFuncs.deseqDE(countTable, sfTable, labels=labels, 
                                    colNames=np.arange(countTable.shape[0]))
        minpctM = np.mean(countTable[labels == eq[0]] > 0.5, axis=0) > max(0.1, 1.99/(labels == eq[0]).sum())
        minpctP = np.mean(countTable[labels == eq[1]] > 0.5, axis=0) > max(0.1, 1.99/ (labels == eq[1]).sum())
        minpct = minpctM | minpctP
        DE = (DEtab["padj"] < 0.05) & (np.abs(DEtab["log2FoldChange"]) > 0.25) & minpct
        results[nzCounts, j, i] = DE
        resT = ss.ttest_ind(res1[subsample1][:, nzCounts], res2[subsample2][:, nzCounts], axis=0,
                    alternative="two-sided")
        resultsTtest[nzCounts, j, i] = fdrcorrection(resT[1])[0] & (np.abs(DEtab["log2FoldChange"]) > 0.25) & minpct
        print(results)
print(rsize)
print(results)
# %%
detectedDESEQ = results.sum(axis=0).mean(axis=1)
detectedT = resultsTtest.sum(axis=0).mean(axis=1)
# %%
# Compute precision-recall
precisions = np.zeros((len(allPts), reps))
recalls = np.zeros((len(allPts), reps))
for i in range(len(allPts)):
    for j in range(reps):
        precisions[i, j] = np.sum(results[:, i, j] & resultsTtest[:, i, j])/np.sum(1e-9+resultsTtest[:, i, j])
        recalls[i, j] = np.sum(results[:, i, j] & resultsTtest[:, i, j])/np.sum(1e-9+results[:, i, j])
meanPrecisions = np.mean(precisions, axis=1)
sdPrecisions = np.std(precisions, axis=1)
meanRecalls = np.mean(recalls, axis=1)
sdRecalls = np.std(recalls, axis=1)
# %%
plt.plot(allPts, detectedDESEQ[:len(allPts)])
plt.plot(allPts, detectedT[:len(allPts)])
plt.ylabel("Number of detected Pol II probes")
plt.xlabel("Number of samples per group")
plt.legend(["Detected with DESeq2", "Detected with t-test"])
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq_heart_DE/t_vs_deseq2.pdf")
plt.show()
# %%
plt.plot(allPts, meanPrecisions)
plt.plot(allPts, meanRecalls)
plt.ylabel("")
plt.xlabel("Number of samples per group")
plt.legend(["Precision", "Recall"])
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq_heart_DE/precisionRecall_t.pdf")
plt.show()
# %%
