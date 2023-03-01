# %%
import os
import sys
from joblib.externals.loky import get_reusable_executor

sys.path.append("./")
sys.setrecursionlimit(10000)
import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyranges as pr
import seaborn as sns
import umap
from lib import rnaseqFuncs
from lib.pyGREATglm import pyGREAT
from lib.utils import matrix_utils, plot_utils
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from scipy.cluster import hierarchy
from scipy.stats import mannwhitneyu, ttest_ind
from settings import params, paths
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import fdrcorrection
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", header=None, sep="\t")
consensuses.columns = ["Chromosome", "Start", "End", "Name", "Score", "Strand", "ThickStart", "ThickEnd"]

chrFile = pd.read_csv(paths.genomeFile, sep="\t", index_col=0, header=None)
sortedIdx = ["chr1", 'chr2','chr3','chr4','chr5','chr6',
              'chr7','chr8','chr9', 'chr10', 'chr11','chr12','chr13','chr14','chr15','chr16','chr17',
              'chr18','chr19','chr20','chr21','chr22','chrX','chrY']
chrFile = chrFile.loc[sortedIdx]

enricher = pyGREAT(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)

# %%
allAnnots = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
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
for case in cases:
    print("case")
    # Select only relevant files and annotations
    annotation = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv", 
                            sep="\t", index_col=0)
    annotation = annotation[annotation["project_id"] == case]
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
        continue
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
    annotation = annotation.loc[order]
    for a in annotation["Sample Type"]:
        if a == "Solid Tissue Normal":
            labels.append(0)
        else:
            labels.append(1)
    labels = np.array(labels)
    if len(labels) < 10 or np.any(np.bincount(labels) < 10):
        print(case, "not enough samples")
        continue
    try:
        os.mkdir(paths.outputDir + "rnaseq/TumorVsNormal/" + case)
    except FileExistsError:
        pass
    allReads = np.array(allReads)
    allCounts = np.concatenate(counts, axis=1).T
    counts = allCounts
    # Remove undected Pol II probes
    nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=3)
    countsNz = allCounts[:, nzCounts]
    studiedConsensusesCase[case] = nzCounts
    # Scran normalization
    sf = rnaseqFuncs.scranNorm(countsNz)
    # ScTransform-like transformation and deviance-based variable selection
    countModel = rnaseqFuncs.RnaSeqModeler().fit(countsNz, sf)
    get_reusable_executor().shutdown(wait=False)
    residuals = countModel.residuals
    countsNorm = countModel.normed
    hv = countModel.hv
    # Compute PCA on the residuals
    decomp = rnaseqFuncs.permutationPA_PCA(residuals[:, hv], mincomp=2)
    # Plot PC 1 and 2
    plt.figure(dpi=500)
    plt.scatter(decomp[:, 0], decomp[:, 1], c=labels)
    plt.savefig(paths.outputDir + f"rnaseq/TumorVsNormal/" + case + "/PCA.pdf")
    plt.show()
    plt.close()
    # Find DE genes
    pctThreshold = 0.1
    lfcMin = 0.25
    res = ttest_ind(countModel.residuals[labels == 0], countModel.residuals[labels == 1], axis=0)[1]
    sig, padj = fdrcorrection(res)
    minpctM = np.mean(countsNz[labels == 1] > 0.5, axis=0) > max(0.1, 1.5/labels.sum())
    minpctP = np.mean(countsNz[labels == 0] > 0.5, axis=0) > max(0.1, 1.5/(1-labels).sum())
    minpct = minpctM | minpctP
    fc = np.mean(countModel.normed[labels == 1], axis=0) / (1e-9+np.mean(countModel.normed[labels == 0], axis=0))
    lfc = np.abs(np.log2(fc)) > lfcMin
    sig = sig & lfc & minpct
    print(sig.sum())

    allDE = consensuses[nzCounts][sig]
    allDE["Score"] = -np.log10(padj[sig])
    allDE.to_csv(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/allDE.bed", sep="\t", header=None, index=None)
    
    delta = np.mean(countModel.residuals[labels == 1], axis=0) - np.mean(countModel.residuals[labels == 0], axis=0)
    allWithScore = consensuses.copy()[["Chromosome", "Start", "End", "Name"]]
    allWithScore["logFDR"] = 0.0
    allWithScore["logFDR"][nzCounts] = -np.log10(padj)
    allWithScore["DeltaRes"] = 0.0
    allWithScore["DeltaRes"][nzCounts] = delta
    allWithScore["DeltaRes"] = 0.0
    allWithScore["DeltaRes"][nzCounts] = delta
    allWithScore["LFC"] = 0.0
    allWithScore["LFC"][nzCounts] = np.log2(fc)
    allWithScore.to_csv(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/allWithStats.bed", sep="\t", index=None)
    enrichedBP = enricher.findEnriched(consensuses[nzCounts][sig], consensuses)
    enricher.plotEnrichs(enrichedBP, savePath=paths.outputDir + f"rnaseq/TumorVsNormal/{case}/DE_greatglm.pdf")
    if len(enrichedBP) > 1:
        enricher.clusterTreemap(enrichedBP, output=paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/DE_great_revigo.pdf")
    enrichedBP.to_csv(paths.outputDir + f"rnaseq/TumorVsNormal/" + case + "/DE_greatglm.tsv", sep="\t")
    # Manhattan plot
    orderP = np.argsort(res)[::-1]
    threshold = -np.log10(res[orderP][np.searchsorted(padj[orderP] < 0.05, True)])
    fig, ax = plot_utils.manhattanPlot(consensuses[nzCounts], chrFile, res, es=None, threshold=threshold)
    fig.savefig(paths.outputDir + f"rnaseq/TumorVsNormal/" + case + "/manhattan_DE.pdf")
    # Plot heatmap and dendrograms
    rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, "correlation")
    colOrder, colLink = matrix_utils.threeStagesHClinkage(residuals.T, "correlation")
    plt.figure(dpi=500)
    hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/dendrogram_col.pdf")
    plt.close()
    # Plot dendrograms
    plt.figure(dpi=500)
    hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1, orientation="left")
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/dendrogram_row.pdf")
    plt.close()
    clippedSQ= np.log(countsNorm+1)
    plot_utils.plotHC(residuals.T, np.array(["Cancer","Normal"])[1-labels], countsNorm.T,  
                    rowOrder=rowOrder, colOrder=colOrder, cmap="vlag", rescale="3SD")
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/heatmap.pdf")
    plt.close()
    # Plot heatmap and dendrograms (hv)
    colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.residuals.T[hv], "correlation")
    plt.figure(dpi=500)
    hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/dendrogram_col_hv.pdf")
    plt.close()
    # Plot dendrograms
    plt.figure(dpi=500)
    hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1, orientation="left")
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/dendrogram_row_hv.pdf")
    clippedSQ= np.log(countsNorm+1)
    plot_utils.plotHC(residuals.T[hv], np.array(["Cancer","Normal"])[1-labels], countsNorm.T[hv],  
                    rowOrder=rowOrder, colOrder=colOrder, cmap="vlag", rescale="3SD")
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/heatmap_hv.pdf")
    plt.close()
    # Plot heatmap and dendrograms (DE)
    decompDE = rnaseqFuncs.permutationPA_PCA(residuals[:, sig], mincomp=2)
    rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decompDE, "correlation")
    colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.residuals.T[sig], "correlation")
    plt.figure(dpi=500)
    hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/dendrogram_col_DE.pdf")
    plt.close()
    # Plot dendrograms
    plt.figure(dpi=500)
    hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1, orientation="left")
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/dendrogram_row_DE.pdf")
    plt.close()
    clippedSQ = countModel.residuals
    plot_utils.plotHC(residuals.T[sig], np.array(["Cancer","Normal"])[1-labels], countsNorm.T[sig],  
                    rowOrder=rowOrder, colOrder=colOrder, cmap="vlag", rescale="3SD")
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/heatmap_DE.pdf")
    plt.close()
    # Plot DE signal
    orderCol = np.argsort(labels)
    meanNormal = np.mean(residuals[labels == 0][:, sig], axis=0)
    meanTumor = np.mean(residuals[labels == 1][:, sig], axis=0)
    diff = meanTumor - meanNormal
    orderRow = np.argsort(diff)
    # Only DE
    plt.figure(dpi=500)
    plot_utils.plotDeHM(residuals[:, sig][orderCol][:, orderRow], labels[orderCol], np.ones(np.sum(sig)))
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/DE_plot_only_DE.pdf")
    plt.close()
    # All Pol II
    orderCol = np.argsort(labels)
    meanNormal = np.mean(residuals[labels == 0], axis=0)
    meanTumor = np.mean(residuals[labels == 1], axis=0)
    diff = meanTumor - meanNormal
    orderRow = np.argsort(diff)
    plt.figure(dpi=500)
    plot_utils.plotDeHM(residuals[orderCol][:, orderRow], labels[orderCol], (sig).astype(float)[orderRow])
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/DE_plot.pdf")
    # plt.show()
    plt.close()
    perCancerDE[case] = np.zeros(allCounts.shape[1])
    perCancerDE[case][nzCounts] = np.where(sig, np.sign(diff), 0.0)
    upreg = consensuses[nzCounts][sig & (diff > 0)]
    upreg["Score"] = -np.log10(padj[sig & (diff > 0)])
    upreg.to_csv(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/DE_upreg.bed", sep="\t", header=None, index=None)
    downreg = consensuses[nzCounts][sig & (diff < 0)]
    downreg["Score"] = -np.log10(padj[sig & (diff < 0)])
    downreg.to_csv(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/DE_downreg.bed", sep="\t", header=None, index=None)
    # UMAP plot
    # Plot UMAP of samples for visualization
    embedding = umap.UMAP(n_neighbors=30, min_dist=0.3,
                        random_state=42, low_memory=False, metric="correlation").fit_transform(decomp)
    plt.figure(figsize=(10,10), dpi=500)
    plt.title(f"{case} samples")
    palette, colors = plot_utils.getPalette(labels)
    plot_utils.plotUmap(embedding, colors)
    patches = []
    for i in np.unique(labels):
        legend = Patch(color=palette[i], label=["Normal", "Cancer"][i])
        patches.append(legend)
    plt.legend(handles=patches, prop={'size': 7})
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/UMAP_samples.pdf")
    plt.show()
    plt.close()
    # Predictive model on PCA space
    predictions = np.zeros(len(labels), dtype=int)
    for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(decomp, labels):
        # Fit power transform on train data only
        x_train = decomp[train]
        # Fit classifier train data
        model = catboost.CatBoostClassifier(class_weights=len(labels) / (2 * np.bincount(labels)), 
                                            random_state=42)
        model.fit(x_train, labels[train], silent=True)
        # Predict on test data
        x_test = decomp[test]
        predictions[test] = model.predict(x_test)
    with open(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/" +  f"classifier_{case}.txt", "w") as f:
        print("Weighted accuracy :", balanced_accuracy_score(labels, predictions), file=f)
        wAccs[case] = [balanced_accuracy_score(labels, predictions)]
        recalls[case] = [recall_score(labels, predictions)]
        precisions[case] = [precision_score(labels, predictions)]
        print("Recall :", recall_score(labels, predictions), file=f)
        print("Precision :", precision_score(labels, predictions), file=f)
        df = pd.DataFrame(confusion_matrix(labels, predictions))
        df.columns = ["Normal Tissue True", "Tumor True"]
        df.index = ["Normal Tissue predicted", "Tumor predicted"]
        print(df, file=f)
# %%
# Summary predictive metrics plots
def plotMetrics(summaryTab, metricName):
    plt.figure(dpi=500)
    order = np.argsort(summaryTab).values[0]
    plt.barh(np.arange(len(summaryTab.T)),summaryTab.values[0][order])
    plt.yticks(np.arange(len(summaryTab.columns)), summaryTab.columns[order])
    plt.xticks(np.linspace(0,1.0,11))
    plt.xlabel(metricName)
    plt.ylabel("Cancer type")
    # plt.vlines(0.5, plt.ylim()[0], plt.ylim()[1], color="red", linestyles="dashed")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
plotMetrics(wAccs, "Balanced accuracy")
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/predictiveWeightedAccuracies.pdf", bbox_inches="tight")
plt.close()
plotMetrics(recalls, "Recall")
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/predictiveRecalls.pdf", bbox_inches="tight")
plt.close()
plotMetrics(precisions, "Precision")
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/predictivePrecisions.pdf", bbox_inches="tight")
plt.close()
# %%
# Plot # of DE Pol II Cancer vs Normal
DEperCancer = pd.DataFrame(np.sum(np.abs(perCancerDE), axis=0)).T
# Normal scale
plt.figure(figsize=(6,4), dpi=500)
order = np.argsort(DEperCancer).values[0]
plt.barh(np.arange(len(DEperCancer.T)),DEperCancer.values[0][order])
plt.yticks(np.arange(len(DEperCancer.columns)), DEperCancer.columns[order])
plt.xlabel("# of DE Pol II Cancer vs Normal")
plt.ylabel("Cancer type")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/DE_countPerCancer.pdf", bbox_inches="tight")
plt.show()
plt.close()
# Log Scale
plt.figure(figsize=(6,4), dpi=500)
order = np.argsort(DEperCancer).values[0]
plt.xscale("symlog")
plt.gca().minorticks_off()
plt.barh(np.arange(len(DEperCancer.T)),DEperCancer.values[0][order])
plt.yticks(np.arange(len(DEperCancer.columns)), DEperCancer.columns[order])
plt.xlabel("# of DE Pol II Cancer vs Normal")
plt.ylabel("Cancer type")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/DE_countPerCancer_log.pdf", bbox_inches="tight")
plt.show()
plt.close()

# %%
# Plot "multi-DE"ness of Pol II et estimate "multiDEness" threshold
cases = ["DE", "Up regulated", "Down regulated"]
for c in cases:
    if c == "DE":
        mat = np.abs(perCancerDE).astype("int32")
    elif c == "Up regulated":
        mat = np.clip(perCancerDE,0,1).astype("int32")
    else:
        mat = np.clip(-perCancerDE,0,1).astype("int32")
    consensusDECount = np.sum(mat, axis=1).astype("int32")
    # Compute null distribution of DE Pol II
    countsRnd = np.zeros(mat.shape[1]+1)
    nPerm = 100
    for i in range(nPerm):
        shuffledDF = np.zeros_like(mat)
        for j, cancer in enumerate(mat.columns):
            # Permute only on detected Pol II
            shuffledDF[:, j] = np.random.permutation(mat.iloc[:, j])
        s = np.sum(shuffledDF, axis=1)
        counts = np.bincount(s)/nPerm
        countsRnd[:counts.shape[0]] += counts
    countsObs = np.bincount(np.sum(mat, axis=1))
    for threshold in range(len(countsObs)):
        randomSum = np.sum(countsRnd[threshold:])
        fpr = randomSum / (np.sum(countsObs[threshold:])+randomSum)
        print(threshold, fpr)
        if fpr < 0.05:
            break
    studied = np.sum(mat, axis=1) >= threshold
    plt.figure(figsize=(6,4), dpi=300)
    plt.hist(consensusDECount, np.arange(consensusDECount.max()+1.5))
    plt.hist(s, np.arange(consensusDECount.max()+1.5), alpha=0.5)
    plt.xlabel(f"{c} in x cancers")
    plt.ylabel("# of Pol II")
    plt.legend(["Observed", "Expected"])
    plt.vlines(threshold + 0.0, plt.ylim()[0], plt.ylim()[1], color="red", linestyles="dashed")
    plt.text(threshold + 0.25, plt.ylim()[1]*0.5 + plt.ylim()[0]*0.5, "FPR < 5%", color="red")
    plt.xticks(np.arange(0,consensusDECount.max()+1)+0.5, np.arange(0,consensusDECount.max()+1))
    plt.savefig(paths.outputDir + f"rnaseq/TumorVsNormal/multiple_{c}.pdf", bbox_inches="tight")
    plt.show()
    # Barplot obs
    plt.figure(figsize=(6,1.0), dpi=300)
    counts0 = countsRnd[0]
    counts12 = np.sum(countsRnd[1:3])
    counts37 = np.sum(countsRnd[3:8])
    counts7plus = np.sum(countsRnd[8:])
    cols = sns.color_palette("vlag", as_cmap=True)(np.linspace(0,1,4))
    plt.barh("Random", counts0, color=cols[0])
    plt.barh("Random", counts12, left=counts0, color=cols[1])
    plt.barh("Random", counts37, left=counts0+counts12, color=cols[2])
    plt.barh("Random", counts7plus, left=counts0+counts12+counts37, color=cols[3])
    counts0 = countsObs[0]
    counts12 = np.sum(countsObs[1:3])
    counts37 = np.sum(countsObs[3:8])
    counts7plus = np.sum(countsObs[8:])
    plt.barh("Observed", counts0, color=cols[0])
    plt.barh("Observed", counts12, left=counts0, color=cols[1])
    plt.barh("Observed", counts37, left=counts0+counts12, color=cols[2])
    plt.barh("Observed", counts7plus, left=counts0+counts12+counts37, color=cols[3])
    plt.legend(["0", "1-2", "3-7", "7+"], loc='upper center', bbox_to_anchor=(0.5, 1.5),
          ncol=4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlim([0,len(consensuses)*1.05])
    plt.savefig(paths.outputDir + f"rnaseq/TumorVsNormal/multiple_{c}_barplot.pdf", bbox_inches="tight")
    plt.show()
    plt.close()
    globallyDEs = consensuses[studied]
    globallyDEs[4] = np.sum(mat, axis=1)
    globallyDEs.to_csv(paths.outputDir + f"rnaseq/TumorVsNormal/globally_{c}.bed", sep="\t", header=None, index=None)
    enrichs = enricher.findEnriched(consensuses[studied], consensuses)
    enricher.plotEnrichs(enrichs, savePath=paths.outputDir + f"rnaseq/TumorVsNormal/globally_{c}_GREAT.pdf")
    enrichs.to_csv(paths.outputDir + f"rnaseq/TumorVsNormal/globally_{c}_GREAT.csv", sep="\t")
    if len(enrichs) > 0:
        enricher.clusterTreemap(enrichs, output=paths.outputDir + "rnaseq/TumorVsNormal/" + f"/globally_{c}_GREAT_revigo.pdf.pdf")

# %%

