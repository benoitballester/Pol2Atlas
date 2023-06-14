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
from lib.utils import matrix_utils, plot_utils, utils
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
# %%
# Setup cancer types
allAnnots = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
try:
    os.mkdir(paths.outputDir + "rnaseq/TumorVsNormal/")
except FileExistsError:
    pass
pvalues = pd.DataFrame()
foldChanges = pd.DataFrame()
recalls = pd.DataFrame()
precisions = pd.DataFrame()
# %%
matched_tissues = {"TCGA-BLCA":"Bladder",
                "TCGA-BRCA":"Breast",
                "TCGA-COAD":"Colon",
                "TCGA-ESCA":"Esophagus",
                "TCGA-KICH":"Kidney",
                "TCGA-KIRC":"Kidney",
                "TCGA-KIRP":"Kidney",
                "TCGA-LIHC":"Liver",
                "TCGA-LUAD":"Lung",
                "TCGA-LUSC":"Lung",
                "TCGA-PRAD":"Prostate",
                "TCGA-READ":"Colon",
                "TCGA-STAD":"Stomach",
                "TCGA-THCA":"Thyroid",
                "TCGA-UCEC":"Uterus"}
# %%
for case in matched_tissues:
    print(case)
    # Load tumor data
    # Select only relevant files and annotations
    annotation = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv", 
                            sep="\t", index_col=0)
    annotation = annotation[annotation["project_id"] == case]
    annotation = annotation[annotation["Sample Type"] == "Primary Tumor"]
    dlFiles = os.listdir(paths.countsTCGA + "500centroid/")
    dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
    ids = np.array([f.split(".")[0] for f in dlFiles])
    inAnnot = np.isin(ids, annotation.index)
    ids = ids[inAnnot]
    dlFiles = np.array(dlFiles)[inAnnot]
    annotation = annotation.loc[ids]
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
    counts = np.concatenate(counts, axis=1).T
    # Load normal GTEx data
    countDir = paths.countsGTEx
    annotation_gtex = pd.read_csv(paths.gtexData + "/tsvs/sample_annot.tsv", 
                            sep="\t", index_col="specimen_id")
    annotation_gtex = annotation_gtex[annotation_gtex["tissue_type"]==matched_tissues[case]]

    dlFiles = os.listdir(countDir + "BG/")
    dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
    ids = np.array([f.split(".")[0] for f in dlFiles])
    inAnnot = np.isin(ids, annotation_gtex.index)
    ids = ids[inAnnot]
    dlFiles = np.array(dlFiles)[inAnnot]
    counts_gtex = []
    allReads = []
    order = []
    allStatus = []
    for f in dlFiles:
        try:
            id = ".".join(f.split(".")[:-3])
            status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                                    header=None, index_col=0, sep="\t", skiprows=1).T
            counts_gtex.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values.astype("int32"))
            allStatus.append(status)
            status = status.drop("Unassigned_Unmapped", axis=1)
            allReads.append(status.values.sum())
            order.append(f.split(".")[0])
        except:
            print(f, "missing")
            continue
    counts_gtex = np.concatenate(counts_gtex, axis=1).T
    # Concatenate datasets
    counts = np.concatenate([counts_gtex, counts], axis=0)
    labels = ["Normal_gtex"] * len(counts_gtex) + ["Tumour_TCGA"] * (len(counts)-len(counts_gtex))
    # Transform counts
    nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
    counts = counts[:, nzCounts]
    # Scran normalization
    sf = rnaseqFuncs.scranNorm(counts)
    # ScTransform-like transformation and deviance-based variable selection
    countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf)
    get_reusable_executor().shutdown(wait=False)
    residuals = countModel.residuals
    countsNorm = countModel.normed
    hv = countModel.hv
    # Compute PCA on the residuals
    decomp = rnaseqFuncs.permutationPA_PCA(residuals[:, hv], perm=1, mincomp=2, max_rank=int(np.min(residuals.shape)/4+1))
    # Plot PC 1 and 2
    binLabels, ref = pd.factorize(labels)
    plt.figure(dpi=500)
    plt.scatter(decomp[:, 0], decomp[:, 1], c=binLabels)
    plt.show()
    plt.close()
    # Find DE genes
    pctThreshold = 0.1
    lfcMin = 0.25
    res = ttest_ind(countModel.residuals[binLabels == 0], countModel.residuals[binLabels == 1], axis=0)[1]
    sig, padj = fdrcorrection(res)
    minpctM = np.mean(counts[binLabels == 1] > 0.5, axis=0) > max(0.1, 1.5/binLabels.sum())
    minpctP = np.mean(counts[binLabels == 0] > 0.5, axis=0) > max(0.1, 1.5/(1-binLabels).sum())
    minpct = minpctM | minpctP
    fc = np.mean(countModel.normed[binLabels == 1], axis=0) / (1e-9+np.mean(countModel.normed[binLabels == 0], axis=0))
    lfc = np.abs(np.log2(fc)) > lfcMin
    ref = pd.read_csv(paths.outputDir + f"rnaseq/TumorVsNormal/{case}/allDE.bed", 
                    header=None, sep="\t")
    probe_id_ref = ref[3].values
    # Set enrichment
    from scipy.stats import hypergeom
    topK = len(probe_id_ref)
    probe_id_gtex = consensuses["Name"].values[nzCounts][np.argsort(res)][:topK]
    marker_intersect = np.intersect1d(probe_id_gtex, probe_id_ref)
    recall = len(marker_intersect) / len(probe_id_ref)
    precision = len(marker_intersect) / len(probe_id_gtex)
    fc = (len(marker_intersect)/len(probe_id_gtex)) / (len(probe_id_ref)/len(consensuses))
    p = hypergeom.sf(len(marker_intersect)-1, len(consensuses), len(probe_id_ref), len(probe_id_gtex))
    print(recall, precision, fc, p)
    recalls[case] = [recall]
    pvalues[case] = [p]
    foldChanges[case] = [fc]
# %%
utils.createDir(paths.outputDir + "rnaseq/TumorVsGTex/")
# %%
import seaborn as sns
plt.figure(dpi=500)
sns.barplot(recalls, orient="h", palette="tab20")
plt.xlabel("Marker Recall")
plt.ylabel("Dataset")
plt.savefig(paths.outputDir + "rnaseq/TumorVsGTex/recalls.pdf", bbox_inches="tight")
# %%
import seaborn as sns
plt.figure(dpi=500)
sns.barplot(foldChanges, orient="h", palette="tab20")
plt.xlabel("Marker concordance fold change against random choice")
plt.ylabel("Dataset")
plt.xlim(1, plt.xlim()[1])
plt.xticks([1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10])
plt.savefig(paths.outputDir + "rnaseq/TumorVsGTex/fc.pdf", bbox_inches="tight")
# %%
