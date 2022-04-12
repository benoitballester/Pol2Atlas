# %%
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("./")
import matplotlib.pyplot as plt
from settings import params, paths
from lib import rnaseqFuncs, normRNAseq, glm
from lib.utils import plot_utils, matrix_utils
from matplotlib.patches import Patch
from scipy.stats import rankdata, chi2, mannwhitneyu, ttest_ind
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,balanced_accuracy_score
import catboost
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
scran = importr("scran")
deseq = importr("DESeq2")
base = importr("base")
s4 = importr("S4Vectors")
from matplotlib.ticker import FormatStrFormatter
import KDEpy
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import scipy.interpolate as si
from scipy.cluster import hierarchy
# %%
from lib.pyGREATglm import pyGREAT
import pyranges as pr
enricher = pyGREAT(oboFile=paths.GOfolder + "/go_eq.obo", geneFile=paths.gencode, 
                   geneGoFile=paths.GOfolder + "/goa_human.gaf")
# %%
allAnnots = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
try:
    os.mkdir(paths.outputDir + "rnaseq/TumorVsNormal2/")
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
    print(case)
    # Select only relevant files and annotations
    annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                            sep="\t", index_col=0)
    annotation = annotation[annotation["project_id"] == case]
    dlFiles = os.listdir(paths.countDirectory + "500centroid/")
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
            status = pd.read_csv(paths.countDirectory + "500centroid/" + fid + ".counts.summary",
                                header=None, index_col=0, sep="\t", skiprows=1).T
            counts.append(pd.read_csv(paths.countDirectory + "500centroid/" + f, header=None, skiprows=2).values)
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
        os.mkdir(paths.outputDir + "rnaseq/TumorVsNormal2/" + case)
    except FileExistsError:
        pass
    allReads = np.array(allReads)
    allCounts = np.concatenate(counts, axis=1).T
    
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    scran = importr("scran")
    counts = allCounts
    # Remove undected Pol II probes
    nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=2)
    countsNz = allCounts[:, nzCounts]
    studiedConsensusesCase[case] = nzCounts
    # Normalize only on the top decile of means to trim very small counts
    # Small counts are usually more represented in samples with larger library sizes,
    # moving down the median and artificially decreasing their size factor
    means = np.mean(countsNz/np.mean(countsNz, axis=1)[:, None], axis=0)
    detected = [np.sum(counts >= i, axis=0) for i in range(20)][::-1]
    mostDetected = np.lexsort(detected)[::-1][:int(counts.shape[1]*0.05+1)]
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        sf = scran.calculateSumFactors(counts.T[mostDetected])
    countModel = rnaseqFuncs.RnaSeqModeler().fit(countsNz, sf)
    anscombe = countModel.anscombeResiduals
    countsNorm = countModel.normed
    pDev, nonOutliers = countModel.hv_selection()
    hv = fdrcorrection(pDev)[0] & nonOutliers
    lv = fdrcorrection(1-pDev)[0] & nonOutliers
    countsNorm = countModel.normed
    # Compute PCA on the anscombe residuals
    # Anscombe residuals have asymptotic convergence to a normal distribution with a null NB distribution
    # Pearson and raw residuals are skewed for non-normal models like the NB or Poisson
    # The number of components is selected with a permutation test
    from lib.jackstraw.permutationPA import permutationPA_PCA
    decomp = permutationPA_PCA(anscombe[:, hv])
    # Plot PC 1 and 2
    plt.figure(dpi=500)
    plt.scatter(decomp[:, 0], decomp[:, 1], c=labels)
    plt.savefig(paths.outputDir + f"rnaseq/TumorVsNormal2/" + case + "/PCA.pdf")
    plt.show()
    plt.close()
    # Find DE genes using DESeq2
    countTable = pd.DataFrame(countsNz.T, columns=order)
    infos = pd.DataFrame(np.array(["N", "T"])[labels], index=order, columns=["Type"])
    infos["sizeFactor"] = sf.ravel()
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        dds = deseq.DESeqDataSetFromMatrix(countData=countTable, colData=infos, design=ro.Formula("~Type"))
        dds = deseq.DESeq(dds, fitType="local")
        res = deseq.results(dds)
        res = pd.DataFrame(res.slots["listData"], index=res.slots["listData"].names).T
        res["padj"] = np.nan_to_num(res["padj"], nan=1.0)
    allDE = consensuses[nzCounts][res["padj"].values < 0.05]
    allDE[4] = -np.log10(res["padj"].values[res["padj"].values < 0.05])
    allDE.to_csv(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/allDE.bed", sep="\t", header=None, index=None)
    # enrichedBP = enricher.findEnriched(consensuses[nzCounts][res["padj"].values < 0.05], consensuses[nzCounts])
    # enricher.plotEnrichs(enrichedBP, savePath=paths.outputDir + f"rnaseq/TumorVsNormal/{case}/DE_greatglm.pdf")
    # enrichedBP.to_csv(paths.outputDir + f"rnaseq/TumorVsNormal2/" + case + "/DE_greatglm.tsv", sep="\t")
    # Plot heatmap and dendrograms
    rowOrder, rowLink = matrix_utils.twoStagesHClinkage(decomp)
    colOrder, colLink = matrix_utils.threeStagesHClinkage(anscombe.T, "correlation")
    plt.figure(dpi=500)
    hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/dendrogram_col.pdf")
    plt.close()
    # Plot dendrograms
    plt.figure(dpi=500)
    hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1, orientation="left")
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/dendrogram_row.pdf")
    clippedSQ= np.log(countsNorm+1)
    plot_utils.plotHC(clippedSQ.T, np.array(["Cancer","Normal"])[1-labels], countsNorm.T,  
                    rowOrder=rowOrder, colOrder=colOrder)
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/heatmap.pdf")
    # Plot heatmap and dendrograms (hv)
    colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.anscombeResiduals.T[hv], "correlation")
    plt.figure(dpi=500)
    hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/dendrogram_col_hv.pdf")
    plt.close()
    # Plot dendrograms
    plt.figure(dpi=500)
    hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1, orientation="left")
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/dendrogram_row_hv.pdf")
    clippedSQ= np.log(countsNorm+1)
    plot_utils.plotHC(clippedSQ.T[hv], np.array(["Cancer","Normal"])[1-labels], countsNorm.T[hv],  
                    rowOrder=rowOrder, colOrder=colOrder)
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/heatmap_hv.pdf")
    # Plot heatmap and dendrograms (DE)
    colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.anscombeResiduals.T[res["padj"].values < 0.05], "correlation")
    plt.figure(dpi=500)
    hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/dendrogram_col_DE.pdf")
    plt.close()
    # Plot dendrograms
    plt.figure(dpi=500)
    hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1, orientation="left")
    plt.axis('off')
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/dendrogram_row_DE.pdf")
    clippedSQ= np.log(countsNorm+1)
    plot_utils.plotHC(clippedSQ.T[res["padj"].values < 0.05], np.array(["Cancer","Normal"])[1-labels], countsNorm.T[res["padj"].values < 0.05],  
                    rowOrder=rowOrder, colOrder=colOrder)
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/heatmap_DE.pdf")
    # Plot DE signal
    orderCol = np.argsort(labels)
    meanNormal = np.mean(anscombe[labels == 0][:, res["padj"].values < 0.05], axis=0)
    meanTumor = np.mean(anscombe[labels == 1][:, res["padj"].values < 0.05], axis=0)
    diff = meanTumor - meanNormal
    orderRow = np.argsort(diff)
    # Only DE
    plt.figure(dpi=500)
    plot_utils.plotDeHM(anscombe[:, res["padj"].values < 0.05][orderCol][:, orderRow], labels[orderCol], np.ones(np.sum(res["padj"].values < 0.05)))
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/DE_plot_only_DE.pdf")
    plt.close()
    # All Pol II
    orderCol = np.argsort(labels)
    meanNormal = np.mean(anscombe[labels == 0], axis=0)
    meanTumor = np.mean(anscombe[labels == 1], axis=0)
    diff = meanTumor - meanNormal
    orderRow = np.argsort(diff)
    plt.figure(dpi=500)
    plot_utils.plotDeHM(anscombe[orderCol][:, orderRow], labels[orderCol], (res["padj"].values < 0.05).astype(float)[orderRow])
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/DE_plot.pdf")
    # plt.show()
    plt.close()
    perCancerDE[case] = np.zeros(allCounts.shape[1])
    perCancerDE[case][nzCounts] = np.where(res["padj"].values > 0.05, 0.0, np.sign(diff))
    upreg = consensuses[nzCounts][(res["padj"].values < 0.05) & (diff > 0)]
    upreg[4] = -np.log10(res["padj"].values[(res["padj"].values < 0.05) & (diff > 0)])
    upreg.to_csv(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/DE_upreg.bed", sep="\t", header=None, index=None)
    downreg = consensuses[nzCounts][(res["padj"].values < 0.05) & (diff < 0)]
    downreg[4] = -np.log10(res["padj"].values[(res["padj"].values < 0.05) & (diff < 0)])
    downreg.to_csv(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/DE_downreg.bed", sep="\t", header=None, index=None)
    # UMAP plot
    # Plot UMAP of samples for visualization
    embedding = umap.UMAP(n_neighbors=30, min_dist=0.5,
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
    plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/UMAP_samples.pdf")
    plt.show()
    plt.close()
    # Predictive model on DE Pol II
    predictions = np.zeros(len(labels), dtype=int)
    for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(decomp, labels):
        # Fit power transform on train data only
        x_train = decomp[train]
        # Fit classifier on scaled train data
        model = catboost.CatBoostClassifier(class_weights=len(labels) / (2 * np.bincount(labels)), 
                                            random_state=42)
        model.fit(x_train, labels[train], silent=True)
        # Scale and predict on test data
        x_test = decomp[test]
        predictions[test] = model.predict(x_test)
    with open(paths.outputDir + "rnaseq/TumorVsNormal2/" + case + "/" +  f"classifier_{case}.txt", "w") as f:
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
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/predictiveWeightedAccuracies.pdf", bbox_inches="tight")
plt.close()
plotMetrics(recalls, "Recall")
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/predictiveRecalls.pdf", bbox_inches="tight")
plt.close()
plotMetrics(precisions, "Precision")
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/predictivePrecisions.pdf", bbox_inches="tight")
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
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/DE_countPerCancer.pdf", bbox_inches="tight")
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
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal2/DE_countPerCancer_log.pdf", bbox_inches="tight")
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
    countsRnd = np.zeros(mat.shape[1])
    nPerm = 100
    for i in range(nPerm):
        shuffledDF = np.zeros_like(mat)
        for j, cancer in enumerate(mat.columns):
            # Permute only on detected Pol II
            shuffledDF[studiedConsensusesCase[cancer], j] = np.random.permutation(mat.iloc[studiedConsensusesCase[cancer], j])
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
    plt.hist(consensusDECount, np.arange(consensusDECount.max()+1))
    plt.hist(s, np.arange(consensusDECount.max()+1), alpha=0.5)
    plt.xlabel(f"{c} in x cancers")
    plt.ylabel("# of Pol II")
    plt.legend(["Observed", "Expected"])
    plt.vlines(threshold + 0.0, plt.ylim()[0], plt.ylim()[1], color="red", linestyles="dashed")
    plt.text(threshold + 0.25, plt.ylim()[1]*0.5 + plt.ylim()[0]*0.5, "FPR < 5%", color="red")
    plt.xticks(np.arange(0,consensusDECount.max()+1)+0.5, np.arange(0,consensusDECount.max()+1))
    plt.savefig(paths.outputDir + f"rnaseq/TumorVsNormal2/multiple_{c}.pdf", bbox_inches="tight")
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
    plt.savefig(paths.outputDir + f"rnaseq/TumorVsNormal2/multiple_{c}_barplot.pdf", bbox_inches="tight")
    plt.show()
    plt.close()
    globallyDEs = consensuses[studied]
    globallyDEs[4] = np.sum(mat, axis=1)
    globallyDEs.to_csv(paths.outputDir + f"rnaseq/TumorVsNormal2/globally_{c}.bed", sep="\t", header=None, index=None)
    enrichs = enricher.findEnriched(consensuses[studied], consensuses)
    enricher.plotEnrichs(enrichs, savePath=paths.outputDir + "rnaseq/TumorVsNormal2/globally_{c}_GREAT.pdf")
    enrichs.to_csv(paths.outputDir + f"rnaseq/TumorVsNormal2/globally_{c}_GREAT.csv", sep="\t")
# %%
