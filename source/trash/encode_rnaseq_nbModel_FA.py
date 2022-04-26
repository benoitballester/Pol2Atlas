# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib import normRNAseq, rnaseqFuncs
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
os.environ["CUDA_VISIBLE_DEVICES"]="2"
countDir = "/scratch/pdelangen/projet_these/outputPol2/rnaseq/encode_counts/"
try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/")
except FileExistsError:
    pass
# %%
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/encode_total_rnaseq_annot_0.tsv", 
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
        counts.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except:
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
ann, eq = pd.factorize(annotation.loc[order]["Annotation"])
# %% 
# Plot FPKM expr per annotation
palette = pd.read_csv(paths.polIIannotationPalette, sep=",")
palette = dict([(d["Annotation"], (d["r"],d["g"],d["b"])) for r,d in palette.iterrows()])
fpkmExpr = np.sum(allCounts/allReads[:, None], axis=1)*100
df = pd.DataFrame(data=np.concatenate([fpkmExpr[:, None], annotation.loc[order]["Annotation"].ravel()[:, None]], axis=1), columns=["Percentage of mapped reads", "Annotation"])
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, showfliers=False)
sns.stripplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, dodge=True, 
                edgecolor="black", jitter=1/3, alpha=1.0, s=2, linewidth=0.1)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/pctmapped_per_annot.pdf", bbox_inches="tight")
# %%
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=2)
counts = allCounts[:, nzCounts]

# %%
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
scran = importr("scran")
deseq = importr("DESeq2")
base = importr("base")
detected = [np.sum(counts >= i, axis=0) for i in range(20)][::-1]
topMeans = np.lexsort(detected)[::-1][:int(counts.shape[1]*0.05+1)]
with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
    sf = scran.calculateSumFactors(counts.T[topMeans])
# %%
import tensorflow as tf
import tensorflow_addons as tfa
from lib.AE import nb_loglike
embeddings = 80
np.random.seed(42)

A = counts[:, np.random.choice(counts.shape[1], 10000, replace=False)].astype("float32")
lossFunc = nb_loglike(A.shape[1])
bestAlpha = tf.identity(lossFunc.logalpha)
lossFunc.logalpha = tf.Variable(bestAlpha, dtype=tf.float32)
f0 = np.log(np.mean(A/sf.reshape(-1,1), axis=0)) + np.log(sf.reshape(-1,1))
fx = f0.astype("float32")
Us = []
for i in range(embeddings):
    U = tf.Variable(tf.random.normal([A.shape[1], 1])/100.0, dtype=tf.float32)
    V = tf.Variable(tf.random.normal([1, A.shape[0]])/100.0, dtype=tf.float32)
    optimizer = tfa.optimizers.AdamW(1e-6, 1e-1, amsgrad=True)
    trainable_weights = [U, V]
    bestLoss = (-1, 1000000000)
    for step in range(10000):
        with tf.GradientTape() as tape:
            A_prime = tf.clip_by_value(tf.transpose(tf.matmul(U, V)) + fx, -30, 30)
            loss = lossFunc([A, A_prime])
        if float(loss) - bestLoss[1] < -0.01:
            bestLoss = (step, float(loss))
            bestU = tf.identity(U)
            bestV = tf.identity(V)
            bestAlpha = tf.identity(lossFunc.logalpha)
        else:
            if step - bestLoss[0] > 20:
                break
        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))
        if step % 20 == 0:
            print(f"Training loss at step {step}: {loss:.4f}")
    if bestLoss[0] != 0:  
        fx += tf.transpose(tf.matmul(bestU, bestV))
        Us.append(bestV.numpy().T)
        lossFunc.logalpha = tf.Variable(bestAlpha, dtype=tf.float32)
decomp = np.concatenate(Us, axis=1)
# %%
matrix_utils.looKnnCV(decomp[:, :20], ann, "correlation",30)
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=50, min_dist=0.5, random_state=0, low_memory=False, metric="correlation").fit_transform(decomp)
plt.figure(figsize=(10,10), dpi=500)
annot, palette, colors = plot_utils.applyPalette(annotation.loc[order]["Annotation"],
                                                np.unique(annotation.loc[order]["Annotation"]),
                                                 paths.polIIannotationPalette, ret_labels=True)
plot_utils.plotUmap(embedding, colors)
patches = []
for i, a in enumerate(annot):
    legend = Patch(color=palette[i], label=a)
    patches.append(legend)
plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
# plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/umap_samples.pdf")
plt.show()
plt.close()
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
clippedSQ= np.log(1+countModel.normed)
plot_utils.plotHC(clippedSQ.T[hv & nonOutliers], annotation.loc[order]["Annotation"], (countModel.normed).T[hv & nonOutliers],  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_hvg.pdf")
# %%
colOrderAll, colLinkAll = matrix_utils.threeStagesHClinkage(countModel.anscombeResiduals.T, "correlation")
# Plot dendrograms
plt.figure(dpi=500)
hierarchy.dendrogram(colLinkAll, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_col_dendrogram.pdf")
plt.show()
clippedSQ= np.log(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["Annotation"], (countModel.normed).T,  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrderAll)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM.pdf")
# %%
# Comparison Pol II biotype vs RNA-seq
import pickle
polIIMerger = pickle.load(open(paths.outputDir + "merger", "rb"))
# %%
annotationDf = pd.read_csv(paths.annotationFile, sep="\t", index_col=0)
annotationsP2, eq2 = pd.factorize(annotationDf.loc[polIIMerger.labels]["Annotation"],
                                sort=True)
signalPerCategory = np.zeros((np.max(annotationsP2)+1, len(polIIMerger.embedding[0])))
signalPerAnnot = np.array([np.sum(polIIMerger.matrix[:, i == annotationsP2]) for i in range(np.max(annotationsP2)+1)])
for i in range(np.max(annotationsP2)+1):
    signalPerCategory[i, :] = np.sum(polIIMerger.matrix[:, annotationsP2 == i], axis=1) / signalPerAnnot[i]
signalPerCategory /= np.sum(signalPerCategory, axis=0)
# %%
rnaseqPerCategory = np.zeros((np.max(ann)+1, len(countModel.normed[1])))
for i in range(np.max(ann)+1):
    rnaseqPerCategory[i, :] = np.mean(countModel.normed.T[:, ann == i], axis=1)
rnaseqPerCategory /= np.sum(rnaseqPerCategory, axis=0)
rnaseqPerCategory /= np.sum(rnaseqPerCategory, axis=1)[:, None]
rnaseqPerCategory /= np.sum(rnaseqPerCategory, axis=0)
signalPerCategory = signalPerCategory[:, nzCounts]
# %%
try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/polII_vs_rnaseq/")
except FileExistsError:
    pass
for test in eq2:
    idx1 = list(eq).index(test)
    idx2 = list(eq2).index(test)
    sig1 = signalPerCategory[idx2] + np.random.normal(size=signalPerCategory[idx2].shape)*0.003
    sig2 = rnaseqPerCategory[idx1]
    plt.figure(dpi=500)
    plt.scatter(sig1, sig2, s=0.1, linewidths=0.0)
    plt.xlabel("Pol II probe % of biotype (+gaussian noise to unstack points)")
    plt.ylabel("Fraction of reads in biotype")
    plt.title(test)
    plt.savefig(paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/{test}.pdf")
    plt.close()
# %%
# Boxplot Top 50% pct Pol II vs bottom 50% Pol II Biotype
for test in eq2:
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
    plt.savefig(paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/boxplot_{test}.pdf")
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
    sns.heatmap(resultMat, cmap="vlag", vmin=0.0, vmax=1.0,)
    plt.xlabel("Pol II annotations")
    plt.ylabel("ENCODE annotations")
    plt.title("log2(Reads on probe > 50% Pol Biotype / Reads on probe < 50% Pol Biotype)")
    plt.savefig(savePath, bbox_inches="tight")
    plt.show()
    plt.close()

plotHmPol2vsRnaseq(eq, eq2, signalPerCategory, rnaseqPerCategory,
                   paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/heatmap_fc.pdf")
plotHmPol2vsRnaseq(eq, eq2, signalPerCategory[:, hv & nonOutliers], rnaseqPerCategory[:, hv & nonOutliers],
                   paths.outputDir + f"rnaseq/encode_rnaseq/polII_vs_rnaseq/heatmap_fc_hv.pdf")
# %%
import scipy.stats as ss
clustsPol2 = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt",dtype=int)[nzCounts]
nClusts = np.max(clustsPol2)+1
nAnnots = len(eq)
zScores = np.zeros((nClusts, nAnnots))
filteredMat = (countModel.normed / np.mean(countModel.normed, axis=0))[:, hv & nonOutliers]
clustsPol2 = clustsPol2[hv & nonOutliers]
for i in np.unique(clustsPol2):
    avgPerAnnotInClust = np.mean(rnaseqPerCategory.T[hv & nonOutliers][clustsPol2 == i], axis=0)
    for j in range(nAnnots):
        hasAnnot = np.arange(nAnnots) == j
        zScores[i, j] = np.log(avgPerAnnotInClust[hasAnnot]/np.mean(avgPerAnnotInClust[np.logical_not(hasAnnot)]))
# %%
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
cb = mpl.colorbar.ColorbarBase(plt.gca(), sns.color_palette("vlag", as_cmap=True), norm, orientation='horizontal')
cb.set_label("95th percentile Z-score")
plt.tight_layout()
# plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/signalPerClustPerAnnot_colorbar.pdf")
plt.show()
# %%
rowOrder = np.argsort(ann)
topCat = rnaseqPerCategory.argmax(axis=0)
signalTopCat = -rnaseqPerCategory[(topCat,range(len(topCat)))]
colOrder = np.lexsort((signalTopCat, topCat))
meanNormed = countModel.normed/np.mean(countModel.normed, axis=0)
epsilon = 1/np.nanmax(np.log(meanNormed), axis=0)
clippedSQ= np.log(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["Annotation"], countModel.normed.T,  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_autorank.pdf")
plt.close()
# %%
rowOrder = np.argsort(ann)
topCat = signalPerCategory.argmax(axis=0)
signalTopCat = -signalPerCategory[(topCat,range(len(topCat)))]
top50 = signalTopCat < -0.51
colOrder = np.lexsort((signalTopCat[top50], topCat[top50]))
clippedSQ= np.log(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T[top50], annotation.loc[order]["Annotation"], countModel.normed.T[top50],  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_rank_pol2_top50.pdf")
plt.close()
plt.figure(dpi=500)
plot_utils.plotHC(countModel.anscombeResiduals.T[top50], annotation.loc[order]["Annotation"], countModel.normed.T[top50],  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_rank_pol2_top50_resid.pdf")
plt.close()
# %%
rowOrder = np.argsort(ann)
topCat = signalPerCategory.argmax(axis=0)[hv & nonOutliers]
signalTopCat = -signalPerCategory[:, hv & nonOutliers][(topCat,range(len(topCat)))]
top50 = signalTopCat < -0.51
colOrder = np.lexsort((signalTopCat[top50], topCat[top50]))
clippedSQ= np.log(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(countModel.anscombeResiduals.T[hv & nonOutliers][top50], annotation.loc[order]["Annotation"], countModel.normed.T[hv & nonOutliers][top50],  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_rank_pol2_top50_hv_resid.pdf")
plt.show()
plt.close()
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T[hv & nonOutliers][top50], annotation.loc[order]["Annotation"], countModel.normed.T[hv & nonOutliers][top50],  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_rank_pol2_top50_hv.pdf")
plt.show()
plt.close()
# %%
rowOrder = np.argsort(ann)
topCat = rnaseqPerCategory.argmax(axis=0)
signalTopCat = -rnaseqPerCategory[(topCat,range(len(topCat)))]
colOrder = np.lexsort((signalTopCat, topCat))
meanNormed = countModel.normed/np.mean(countModel.normed, axis=0)
epsilon = 1/np.nanmax(np.log(meanNormed), axis=0)
clippedSQ= np.log(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["Annotation"], polIIMerger.matrix[nzCounts],  
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder, labelsPct=annotationDf.loc[polIIMerger.labels]["Annotation"])
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM_autorank_PolII_signal.pdf")
plt.show()
plt.close()
# %%
