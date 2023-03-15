# %%
import os
import sys
sys.path.append("./")
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

countDir = paths.countsENCODE
try:
    os.mkdir(paths.outputDir + "rnaseq/")
except FileExistsError:
    pass
try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/")
except FileExistsError:
    pass
# %%
annotation = pd.read_csv(paths.encodeAnnot, 
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
        # countsBG.append(pd.read_csv(paths.countDirectory + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values.astype("int32"))
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except:
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
conv = pd.read_csv(paths.tissueToSimplified, sep="\t", index_col="Tissue")
annotation.loc[order, "Annotation"] = conv.loc[annotation.loc[order]["Annotation"].values]["Simplified"].values
annTxt = annotation.loc[order]["Annotation"]
ann, eq = pd.factorize(annTxt)
# %%
try:
    os.mkdir(paths.outputDir + "rnaseq/count_tables/")
except:
    pass
try:
    os.mkdir(paths.outputDir + "rnaseq/count_tables/ENCODE/")
except:
    pass
rnaseqFuncs.saveDataset(allCounts, pd.DataFrame(order), paths.outputDir + "rnaseq/count_tables/ENCODE/")
# %% 
# Plot FPKM expr per annotation
fpkmExpr = np.sum(allCounts/allReads[:, None], axis=1)*100
df = pd.DataFrame(data=np.concatenate([fpkmExpr[:, None], annTxt.ravel()[:, None]], axis=1), columns=["Percentage of mapped reads", "Annotation"])
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(data=df, x="Percentage of mapped reads", y="Annotation", showfliers=False)
sns.stripplot(data=df, x="Percentage of mapped reads", y="Annotation", dodge=True, 
                edgecolor="black", jitter=1/3, alpha=1.0, s=2, linewidth=0.1)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/pctmapped_per_annot.pdf", bbox_inches="tight")
# %%
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=3)
counts = allCounts[:, nzCounts]
# %%
sf = rnaseqFuncs.scranNorm(counts).astype("float32")
# %%

""" try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
except FileExistsError:
    pass
rnaseqFuncs.limma1vsAll(counts, sf, annTxt, np.arange(len(nzCounts))[nzCounts], 
                        paths.outputDir + "rnaseq/encode_rnaseq/DE/") """

# %%
from lib.rnaseqFuncs import RnaSeqModeler

countModel = RnaSeqModeler().fit(counts, sf, figSaveDir=paths.tempDir)
from joblib.externals.loky import get_reusable_executor
get_reusable_executor().shutdown(wait=True)
hv = countModel.hv
# %%
from sklearn.preprocessing import StandardScaler
feat = countModel.residuals[:, hv]
decomp = rnaseqFuncs.permutationPA_PCA(feat, perm=10, max_rank=250, figSaveDir=paths.tempDir)
matrix_utils.looKnnCV(decomp, ann, "correlation", 1)
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=42, low_memory=False, metric="correlation").fit_transform(decomp)

import plotly.express as px

df = pd.DataFrame(embedding, columns=["x","y"])
df["Annotation"] = annotation.loc[order]["Annotation"].values
df["Biosample term name"] = annotation.loc[order]["Biosample term name"].values

annot, palette, colors = plot_utils.applyPalette(annotation.loc[order]["Annotation"],
                                                np.unique(annotation.loc[order]["Annotation"]),
                                                 paths.polIIannotationPalette, ret_labels=True)
palettePlotly = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in palette]
colormap = dict(zip(annot, palettePlotly))     

fig = px.scatter(df, x="x", y="y", color="Annotation", color_discrete_map=colormap,
                hover_data=['Biosample term name'], width=800, height=800)
fig.show()
fig.write_image(paths.outputDir + "rnaseq/encode_rnaseq/umap_samples.pdf")
fig.write_html(paths.outputDir + "rnaseq/encode_rnaseq/umap_samples.pdf" + ".html")
# %%
plt.figure(figsize=(10,10), dpi=500)

plot_utils.plotUmap(embedding, colors)
patches = []
for i, a in enumerate(annot):
    legend = Patch(color=palette[i], label=a)
    patches.append(legend)
plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/umap_samples2.pdf")
plt.show()
plt.close()
# %%
from lib.pyGREATglm import pyGREAT as pyGREATglm

enricherglm = pyGREATglm(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensuses.columns = ["Chromosome", "Start", "End", "Name", "Score", "Strand", "ThickStart", "ThickEnd"]

try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
except FileExistsError:
    pass

# %%
pctThreshold = 0.1
lfcMin = 0.25
for i in np.unique(ann):
    print(eq[i])
    labels = (ann == i).astype(int)
    res2 = ss.ttest_ind(countModel.residuals[ann == i], countModel.residuals[ann != i], axis=0,
                    alternative="greater")
    # res2 = welshTperm(countModel.residuals, ann == i, 10000, alternative="greater", workerPool=pool)
    sig = fdrcorrection(res2[1])[0]
    print(sig.sum())
    delta = np.mean(countModel.residuals[ann == i], axis=0) - np.mean(countModel.residuals[ann != i], axis=0)
    minpct = np.mean(counts[ann == i] > 0.5, axis=0) > max(0.1, 1.5/labels.sum())
    fc = np.mean(countModel.normed[ann == i], axis=0) / (1e-9+np.mean(countModel.normed[ann != i], axis=0))
    lfc = np.log2(fc) > lfcMin
    sig = sig & lfc & minpct
    print(sig.sum())
    res = pd.DataFrame(res2, columns=consensuses.index[nzCounts], index=["stat", "pval"]).T
    res["Upreg"] = 1-sig.astype(int)
    res["Delta"] = -delta
    orderDE = np.lexsort(res[["Delta","pval","Upreg"]].values.T)
    res["Delta"] = delta
    res["Upreg"] = sig.astype(int)
    fname = eq[i].replace("/", "-")
    res = res.iloc[orderDE]
    res.to_csv(paths.outputDir + f"rnaseq/encode_rnaseq/DE/res_{fname}.csv")
    test = consensuses[nzCounts][sig]
    test.to_csv(paths.outputDir + f"rnaseq/encode_rnaseq/DE/bed_{fname}", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
    pvals = enricherglm.findEnriched(test, background=consensuses)
    enricherglm.plotEnrichs(pvals)
    enricherglm.clusterTreemap(pvals, score="-log10(pval)", 
                                output=paths.outputDir + f"rnaseq/encode_rnaseq/DE/great_{fname}.pdf")


# %%
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(feat.T, "correlation")
# %%
# Plot dendrograms
from scipy.cluster import hierarchy

plt.figure(dpi=500)
hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_hvg_col_dendrogram.pdf")
plt.show()
plt.close()
# Plot dendrograms
plt.figure(dpi=500)
hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1, orientation="left")
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_row_dendrogram.pdf")
plt.show()
plt.close()
# %%
clippedSQ = np.log10(1+countModel.normed)
plot_utils.plotHC(clippedSQ.T[hv], annotation.loc[order]["Annotation"], (countModel.normed).T[hv],  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_hvg.pdf")
# %%
colOrderAll, colLinkAll = matrix_utils.threeStagesHClinkage(countModel.residuals.T, "correlation")
# Plot dendrograms
plt.figure(dpi=500)
hierarchy.dendrogram(colLinkAll, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_col_dendrogram.pdf")
plt.show()
clippedSQ = np.log10(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["Annotation"], (countModel.normed).T,  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrderAll)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM.pdf")
# %%
# Comparison Pol II biotype vs RNA-seq
import pickle

polIIMerger = pickle.load(open(paths.outputDir + "merger", "rb"))

mtx = polIIMerger.matrix[nzCounts, :]
annotationDf = pd.read_csv(paths.annotationFile, sep="\t", index_col=0)
annotationsP2, eq2 = pd.factorize(annotationDf.loc[polIIMerger.labels]["Annotation"],
                                sort=True)
signalPerCategory = np.zeros((np.max(annotationsP2)+1, len(mtx)))
signalPerAnnot = np.array([np.sum(mtx[:, i == annotationsP2]) for i in range(np.max(annotationsP2)+1)])
for i in range(np.max(annotationsP2)+1):
    signalPerCategory[i, :] = np.sum(mtx[:, annotationsP2 == i], axis=1) / signalPerAnnot[i]
signalPerCategory /= np.sum(signalPerCategory, axis=0)

rnaseqPerCategory = np.zeros((np.max(ann)+1, len(countModel.normed[1])))
for i in range(np.max(ann)+1):
    rnaseqPerCategory[i, :] = np.mean(countModel.normed.T[:, ann == i], axis=1)
rnaseqPerCategory /= np.sum(rnaseqPerCategory, axis=0)
rnaseqPerCategory /= np.sum(rnaseqPerCategory, axis=1)[:, None]
rnaseqPerCategory /= np.sum(rnaseqPerCategory, axis=0)
# %%

try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/polII_vs_rnaseq/")
except FileExistsError:
    pass
sharedAnnots = np.intersect1d(eq2, eq)
for test in sharedAnnots:
    idx1 = list(eq).index(test)
    idx2 = list(eq2).index(test)
    sig1 = signalPerCategory[idx2] + np.random.normal(size=signalPerCategory[idx2].shape)*0.003
    sig2 = rnaseqPerCategory[idx1]
    plt.figure(dpi=500)
    plt.scatter(sig1, sig2, s=0.1, linewidths=0.0)
    plt.xlabel("Pol II probe % of biotype (+gaussian noise to unstack points)")
    plt.ylabel("Fraction of reads in biotype")
    plt.title(test)
    fname = test.replace("/", "-")
    plt.savefig(paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/{fname}.pdf")
    plt.close()
# %%
# Boxplot Top 50% pct Pol II vs bottom 50% Pol II Biotype
for test in sharedAnnots:
    idx1 = list(eq).index(test)
    idx2 = list(eq2).index(test)
    majoritaryPol2 = signalPerCategory[idx2] > 0.5
    signal = pd.DataFrame(rnaseqPerCategory[idx1], columns=["Fraction of RNA-seq reads in probe"])
    signal["Category"] = np.where(signalPerCategory[idx2] > 0.5, "> 50% of probe biotype", "< 50% of probe biotype")
    upper = rnaseqPerCategory[idx1][signalPerCategory[idx2] > 0.5]
    lower = rnaseqPerCategory[idx1][signalPerCategory[idx2] <= 0.5]
    stat, p = ss.mannwhitneyu(upper, lower)
    fname = test.replace("/", "-")
    plt.figure(dpi=500)
    sns.boxplot(data=signal, x="Category", y="Fraction of RNA-seq reads in probe", showfliers=False)
    sns.stripplot(data=signal, x="Category", y="Fraction of RNA-seq reads in probe", dodge=True, 
                edgecolor="black", jitter=1/4, alpha=1.0, s=0.5)
    plt.title(test + f" (p-value: {p}, direction: {np.sign(np.median(upper)-np.median(lower))})")
    plt.savefig(paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/boxplot_{fname}.pdf")
    plt.close()
# %%
# HM FC enrich Top 50% pct Pol II vs bottom 50% Pol II Biotype
def plotHmPol2vsRnaseq(eq, eq2, signalPolIIperAnnot, signalRnaseqPerAnnot, savePath):
    resultMat = []
    x = []
    for test in eq:
        res_annot_encode = []
        y = []
        for test2 in eq2:
            idx1 = list(eq).index(test)
            idx2 = list(eq2).index(test2)
            upper = signalRnaseqPerAnnot[idx1][signalPolIIperAnnot[idx2] > 0.5]
            lower = signalRnaseqPerAnnot[idx1][signalPolIIperAnnot[idx2] <= 0.5]
            fc = np.nan_to_num(np.log2(np.mean(upper)/np.mean(lower)))
            res_annot_encode.append(fc)
            y.append(test2)   
        x.append(test)
        resultMat.append(res_annot_encode)
    matching = np.isin(x, y)
    resultMat = pd.DataFrame(resultMat, x, y)
    resultMat = resultMat[np.sort(y)]
    resultMat = resultMat.loc[np.sort(np.array(x)[matching])]
    plt.figure(dpi=500)
    sns.heatmap(resultMat, cmap="vlag", vmin=-1.0, vmax=1.0, xticklabels=True, yticklabels=True)
    plt.xlabel("Pol II annotations")
    plt.ylabel("ENCODE annotations")
    plt.title("log2(Reads on probe > 50% Pol Biotype / Reads on probe < 50% Pol Biotype)")
    plt.savefig(savePath, bbox_inches="tight")
    plt.show()
    plt.close()

plotHmPol2vsRnaseq(eq, eq2, signalPerCategory, rnaseqPerCategory,
                   paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/heatmap_fc.pdf")
plotHmPol2vsRnaseq(eq, eq2, signalPerCategory[:, hv], rnaseqPerCategory[:, hv],
                   paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/heatmap_fc_hv.pdf")
# %%
# Cluster-annotation relationship
import scipy.stats as ss

clustsPol2 = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt",dtype=int)[nzCounts]
nClusts = np.max(clustsPol2)+1
nAnnots = len(eq)
zScores = np.zeros((nClusts, nAnnots))
filteredMat = (countModel.normed / np.mean(countModel.normed, axis=0))[:, hv]
clustsPol2 = clustsPol2[hv]
for i in np.unique(clustsPol2):
    avgPerAnnotInClust = np.mean(rnaseqPerCategory.T[hv][clustsPol2 == i], axis=0)
    for j in range(nAnnots):
        hasAnnot = np.arange(nAnnots) == j
        zScores[i, j] = np.log(avgPerAnnotInClust[hasAnnot]/np.mean(avgPerAnnotInClust[np.logical_not(hasAnnot)]))
rowOrder, colOrder = matrix_utils.HcOrder(np.nan_to_num(zScores))
rowOrder = np.loadtxt(paths.outputDir + "clusterBarplotOrder.txt").astype(int)
zClip = zScores
zNorm = np.clip(zClip,0.0,1.0)

plt.figure(dpi=300)
sns.heatmap(zNorm[rowOrder].T[colOrder], cmap="vlag", linewidths=0.1, linecolor='black', cbar=False)
plt.gca().set_aspect(2.0)
plt.yticks(np.arange(len(eq))+0.5, eq[colOrder])
plt.xticks(np.arange(len(rowOrder))+0.5, np.arange(len(rowOrder))[rowOrder], rotation=90, fontsize=6)
plt.xlabel(f"{len(zNorm)} Pol II clusters")
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/signalPerClustPerAnnot.pdf", bbox_inches="tight")
plt.show()
plt.figure(figsize=(6, 1), dpi=300)
norm = mpl.colors.Normalize(vmin=0, vmax=np.percentile(zClip, 95))
cb = mpl.colorbar.ColorbarBase(plt.gca(), cmap=sns.color_palette("vlag", as_cmap=True), norm=norm, orientation='horizontal')
cb.set_label("95th percentile Z-score")
plt.tight_layout()
# plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/signalPerClustPerAnnot_colorbar.pdf")
plt.show()
# %%
# All, encode ordered, encode signal
rowOrder = np.argsort(ann)
topCat = rnaseqPerCategory.argmax(axis=0)
signalTopCat = -rnaseqPerCategory[(topCat,range(len(topCat)))]
colOrder = np.lexsort((signalTopCat, topCat))
meanNormed = countModel.normed/np.mean(countModel.normed, axis=0)
epsilon = 1/np.nanmax(np.log(meanNormed), axis=0)
clippedSQ = np.log10(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["Annotation"], countModel.normed.T,  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/HM_all_encode_order_encode_signal.pdf")
plt.close()
# %%
# At least 50% biotype, Pol II ordered, at least 50% biotype, Pol II signal
rowOrder = np.argsort(ann)
topCat = signalPerCategory.argmax(axis=0)
signalTopCat = -signalPerCategory[(topCat,range(len(topCat)))]
top50 = signalTopCat < -0.51
colOrder = np.lexsort((signalTopCat[top50], topCat[top50]))
clippedSQ = np.log10(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T[top50], annotation.loc[order]["Annotation"],
                  mtx[top50],
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder, 
                  labelsPct=annotationDf.loc[polIIMerger.labels]["Annotation"])
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/HM_top50Pol_pol2_order_pol2_signal.pdf")
plt.close()
# %%
# All, encode ordered, Pol II signal
rowOrder = np.argsort(ann)
topCat = rnaseqPerCategory.argmax(axis=0)
signalTopCat = -rnaseqPerCategory[(topCat,range(len(topCat)))]
colOrder = np.lexsort((signalTopCat, topCat))
meanNormed = countModel.normed/np.mean(countModel.normed, axis=0)
epsilon = 1/np.nanmax(np.log(meanNormed), axis=0)
clippedSQ = np.log10(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["Annotation"],
                  mtx,
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder, 
                  labelsPct=annotationDf.loc[polIIMerger.labels]["Annotation"])
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/HM_all_encode_order_pol2_signal.pdf")
plt.show()
plt.close()
# %%
# All, encode ordered, Pol II signal, min50% pol
rowOrder = np.argsort(ann)
topCat = signalPerCategory.argmax(axis=0)
signalTopCat = -signalPerCategory[(topCat,range(len(topCat)))]
top50 = signalTopCat < -0.51
topCat = rnaseqPerCategory.argmax(axis=0)
signalTopCat = -rnaseqPerCategory[(topCat,range(len(topCat)))]
colOrder = np.lexsort((signalTopCat[top50], topCat[top50]))
meanNormed = countModel.normed/np.mean(countModel.normed, axis=0)
epsilon = 1/np.nanmax(np.log(meanNormed), axis=0)
clippedSQ = np.log10(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T[top50], annotation.loc[order]["Annotation"],
                  mtx[top50],
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder, 
                  labelsPct=annotationDf.loc[polIIMerger.labels]["Annotation"])
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/HM_50Pol_encode_order_pol2_signal.pdf")
plt.show()
plt.close()
# %%
