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

countDir = paths.countsGTEx
try:
    os.mkdir(paths.outputDir + "rnaseq/gtex_rnaseq/")
except FileExistsError:
    pass
# %%
annotation = pd.read_csv(paths.gtexData + "/tsvs/sample.tsv", 
                        sep="\t", index_col="specimen_id")

colors = pd.read_csv(paths.gtexData + "colors.txt", 
                        sep="\t", index_col="tissue_site_detail")
dlFiles = os.listdir(countDir + "BG/")
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
ann, eq = pd.factorize(annotation.loc[order]["tissue_type"])

# %%
nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
counts = counts[:, nzCounts]

# %%
sf = rnaseqFuncs.scranNorm(counts)
# %%
from sklearn.preprocessing import StandardScaler
design = np.ones((len(counts), 1))
ischemicTime = annotation.loc[order]["total_ischemic_time"].fillna(annotation.loc[order]["total_ischemic_time"].median())
ischemicTime = StandardScaler().fit_transform(ischemicTime.values.reshape(-1,1))
design = np.concatenate([design, ischemicTime], axis=1)
countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf, maxThreads=32)
hv = countModel.hv

# %%
feat = countModel.residuals[:, hv]
decomp = rnaseqFuncs.permutationPA_PCA(feat, 1, max_rank=1000, returnModel=False)
matrix_utils.looKnnCV(decomp, ann, "correlation", 5)
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(decomp)
# %%
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
df["Organ"] = annotation.loc[order]["tissue_type"].values
df["Organ detailled"] = annotation.loc[order]["tissue_type_detail"].values
df = df.sample(frac=1)
colormap = dict(zip(colors.index, colors["color_hex"])) 
fig = px.scatter(df, x="x", y="y", color="Organ detailled", color_discrete_map=colormap,
                hover_data=['Organ detailled'], width=1200, height=800)
fig.update_traces(marker=dict(size=3*np.sqrt(len(df)/7500)))
fig.show()
fig.write_image(paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples.pdf")
fig.write_html(paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples.pdf" + ".html")
# %%
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
df["Ischemic time"] = ischemicTime
fig = px.scatter(df, x="x", y="y", color="Ischemic time", width=1200, height=800)
fig.show()

# %%
plt.figure(figsize=(10,10), dpi=500)
palette, colors = plot_utils.getPalette(ann)
plot_utils.plotUmap(embedding, colors)
patches = []
for i, a in enumerate(eq):
    legend = Patch(color=palette[i], label=a)
    patches.append(legend)
plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples_bad.pdf")
plt.show()
plt.close()
# %%
# %%
from lib.pyGREATglm import pyGREAT as pyGREAT
enricher = pyGREAT(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)

try:
    os.mkdir(paths.outputDir + "rnaseq/gtex_rnaseq/DE/")
except FileExistsError:
    pass
# %%
from lib.utils.reusableUtest import mannWhitneyAsymp
tester = mannWhitneyAsymp(countModel.residuals)
# %%
pctThreshold = 0.1
lfcMin = 0.25
for i in np.unique(ann):
    print(eq[i])
    labels = (ann == i).astype(int)
    res2 = tester.test(labels, "less")
    sig = fdrcorrection(res2[1])[0]
    minpct = np.mean(counts[ann == i] > 0.5, axis=0) > max(0.1, 1.5/labels.sum())
    fc = np.mean(counts[ann == i], axis=0) / (1e-9+np.mean(counts[ann != i], axis=0))
    lfc = np.log2(fc) > lfcMin
    sig = sig & lfc & minpct
    print(sig.sum())
    res = pd.DataFrame(res2[::-1], columns=consensuses.index[nzCounts], index=["pval", "stat"]).T
    res["Upreg"] = sig.astype(int)
    res.to_csv(paths.outputDir + f"rnaseq/gtex_rnaseq/DE/res_{eq[i]}.csv")
    test = consensuses[nzCounts][sig]
    test.to_csv(paths.outputDir + f"rnaseq/gtex_rnaseq/DE/bed_{eq[i]}", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
    '''
    pvals = enricher.findEnriched(test, background=consensuses)
    enricher.plotEnrichs(pvals)
    enricher.clusterTreemap(pvals, score="-log10(pval)", 
                                output=paths.outputDir + f"rnaseq/gtex_rnaseq/DE/great_{eq[i]}.pdf")
    '''                           
# %%
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(feat.T, "correlation")
# %%
# Plot dendrograms
from scipy.cluster import hierarchy
plt.figure(dpi=500)
hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/gtex_HM_hvg_col_dendrogram.pdf")
plt.show()
plt.close()
# Plot dendrograms
plt.figure(dpi=500)
hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1, orientation="left")
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/gtex_HM_row_dendrogram.pdf")
plt.show()
plt.close()
# %%
clippedSQ= np.log(1+countModel.normed)
plot_utils.plotHC(clippedSQ.T[hv], annotation.loc[order]["tissue_type"], (countModel.normed).T[hv],  
                  rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/gtex_HM_hvg.pdf", bbox_inches="tight")
# %%
colOrderAll, colLinkAll = matrix_utils.threeStagesHClinkage(countModel.residuals.T, "correlation")
# Plot dendrograms
plt.figure(dpi=500)
hierarchy.dendrogram(colLinkAll, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/gtex_HM_col_dendrogram.pdf")
plt.show()
clippedSQ= np.log(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["tissue_type"], (countModel.normed).T,  
                  rowOrder=rowOrder, colOrder=colOrderAll)
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/gtex_HM.pdf", bbox_inches="tight")

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
    os.mkdir(paths.outputDir + "rnaseq/gtex_rnaseq/polII_vs_rnaseq/")
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
    plt.savefig(paths.outputDir + f"rnaseq/gtex_rnaseq/polII_vs_rnaseq/{test}.pdf")
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
    plt.figure(dpi=500)
    sns.boxplot(data=signal, x="Category", y="Fraction of RNA-seq reads in probe", showfliers=False)
    sns.stripplot(data=signal, x="Category", y="Fraction of RNA-seq reads in probe", dodge=True, 
                edgecolor="black", jitter=1/4, alpha=1.0, s=0.5)
    plt.title(test + f" (p-value: {p}, direction: {np.sign(np.median(upper)-np.median(lower))})")
    plt.savefig(paths.outputDir + f"rnaseq/gtex_rnaseq/polII_vs_rnaseq/boxplot_{test}.pdf")
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
    plt.ylabel("GTEx annotations")
    plt.title("log2(Reads on probe > 50% Pol Biotype / Reads on probe < 50% Pol Biotype)")
    plt.savefig(savePath, bbox_inches="tight")
    plt.show()
    plt.close()

plotHmPol2vsRnaseq(eq, eq2, signalPerCategory, rnaseqPerCategory,
                   paths.outputDir + f"rnaseq/gtex_rnaseq/polII_vs_rnaseq/heatmap_fc.pdf")
plotHmPol2vsRnaseq(eq, eq2, signalPerCategory[:, hv], rnaseqPerCategory[:, hv],
                   paths.outputDir + f"rnaseq/gtex_rnaseq/polII_vs_rnaseq/heatmap_fc_hv.pdf")
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
plt.yticks(np.arange(len(eq))+0.5, eq[colOrder], rotation=0)
plt.xticks(np.arange(len(rowOrder))+0.5, np.arange(len(rowOrder))[rowOrder], rotation=90, fontsize=6)
plt.xlabel(f"{len(zNorm)} Pol II clusters")
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/signalPerClustPerAnnot.pdf", bbox_inches="tight")
plt.show()
plt.figure(figsize=(6, 1), dpi=300)
norm = mpl.colors.Normalize(vmin=0, vmax=np.percentile(zClip, 95))
cb = mpl.colorbar.ColorbarBase(plt.gca(), cmap=sns.color_palette("vlag", as_cmap=True), norm=norm, orientation='horizontal')
cb.set_label("95th percentile Z-score")
plt.tight_layout()
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/signalPerClustPerAnnot_colorbar.pdf")
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
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["tissue_type"], countModel.normed.T,  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/HM_all_gtex_order_encode_signal.pdf")
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
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/HM_top50Pol_pol2_order_pol2_signal.pdf")
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
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/HM_all_encode_order_pol2_signal.pdf")
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
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/HM_50Pol_encode_order_pol2_signal.pdf")
plt.show()
plt.close()