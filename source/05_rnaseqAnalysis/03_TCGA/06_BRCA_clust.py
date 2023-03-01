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

utils.createDir(paths.outputDir + "rnaseq/BRCA_clust")
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
# Select only relevant files and annotations
case = "TCGA-BRCA"
annotation = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
annotation = annotation[(annotation["project_id"] == case)]
dlFiles = os.listdir(paths.countsTCGA + "500centroid/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
ids = np.array([f.split(".")[0] for f in dlFiles])
inAnnot = np.isin(ids, annotation.index) 
inAnnot = inAnnot
ids = ids[inAnnot]
dlFiles = np.array(dlFiles)[inAnnot]
annotation = annotation.loc[ids]
survived = (annotation["vital_status"] == "Alive").values
timeToEvent = annotation["days_to_last_follow_up"].where(survived, annotation["days_to_death"])
# Drop rows with missing survival information
toDrop = timeToEvent.index[timeToEvent == "'--"]
boolIndexing = np.logical_not(np.isin(timeToEvent.index, toDrop))
timeToEvent = timeToEvent.drop(toDrop)
timeToEvent = timeToEvent.astype("float")
survived = survived[boolIndexing]
annotation.drop(toDrop)
dlFiles = dlFiles[boolIndexing]

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
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
counts = allCounts
# Remove undected Pol II probes
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=3)
countsNz = allCounts[:, nzCounts]
studiedConsensusesCase[case] = nzCounts
# Scran normalization
sf = rnaseqFuncs.scranNorm(countsNz)
# %%
# ScTransform-like transformation and deviance-based variable selection
countModel = rnaseqFuncs.RnaSeqModeler().fit(countsNz, sf)
get_reusable_executor().shutdown(wait=False)
residuals = countModel.residuals
countsNorm = countModel.normed
hv = countModel.hv
# Compute PCA on the residuals
decomp = rnaseqFuncs.permutationPA_PCA(residuals, mincomp=2) 
# %%
groups = np.zeros(len(labels),dtype=int)
groups[labels==1] = 1+matrix_utils.graphClustering(decomp[labels==1], "correlation", restarts=10, approx=False)
# Plot PC 1 and 2
plt.figure(dpi=500)
plt.scatter(decomp[:, 0], decomp[:, 1], c=plot_utils.getPalette(groups)[1], s=0.5)
plt.show()
plt.close()
# %%
embedding = umap.UMAP(n_neighbors=30, min_dist=0.3,
                    random_state=42, low_memory=False, metric="correlation").fit_transform(decomp)
plt.figure(figsize=(10,10), dpi=500)
plt.title(f"{case} samples")
pal, cols = plot_utils.getPalette(groups)
plot_utils.plotUmap(embedding, cols)
patches = []
for i in np.unique(groups):
    legend = Patch(color=pal[i], label=str(i))
    patches.append(legend)
plt.legend(handles=patches)
plt.savefig(paths.outputDir + "rnaseq/BRCA_clust/UMAP_clusters.pdf")
# %%
# Plot heatmap and dendrograms (hv)
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.residuals.T[hv], "correlation")
grpStr = groups.astype("str")
grpStr[grpStr != "0"] = ["Tumor_" + c for c in grpStr[grpStr != "0"]                                        ]
grpStr[grpStr == "0"] = "Normal"
plot_utils.plotHC(residuals.T[hv], grpStr, countsNorm.T[hv],  
                rowOrder=rowOrder, colOrder=colOrder, cmap="vlag", rescale="3SD")
plt.savefig(paths.outputDir + "rnaseq/BRCA_clust/HM_clusters.pdf")
# %%
import scipy.stats as ss
pctThreshold = 0.1
lfcMin = 2.0
refGroup = groups == 0
for i in range(1, groups.max()+1):
    print(i)
    res2 = ss.ttest_ind(countModel.residuals[groups == i], countModel.residuals[refGroup], axis=0,
                        alternative="two-sided")
    sig = fdrcorrection(res2[1])[0]
    minpctM = np.mean(countsNz[groups == i] > 0.5, axis=0) > max(0.1, 1.5/(groups == i).sum())
    minpctP = np.mean(countsNz[refGroup] > 0.5, axis=0) > max(0.1, 1.5/(refGroup).sum())
    minpct = minpctM | minpctP
    fc = np.mean(countModel.normed[groups == i], axis=0) / (1e-9+np.mean(countModel.normed[refGroup], axis=0))
    lfc = np.abs(np.log2(fc)) > lfcMin
    print(sig.sum())
    sig = sig & lfc & minpct
    print(sig.sum())
    res = pd.DataFrame(res2[::-1], columns=consensuses.index[nzCounts], index=["pval", "stat"]).T
    res["Upreg"] = sig.astype(int)
    res.to_csv(paths.outputDir + f"rnaseq/BRCA_clust/res_{i}.csv")
    test = consensuses[nzCounts][sig]
    test.to_csv(paths.outputDir + f"rnaseq/BRCA_clust/bed_{i}", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
# %%
# Cox LM
import lifelines as ll
dummy = pd.get_dummies(groups)
df = pd.DataFrame()
df["Dead"] = np.logical_not(survived[np.isin(timeToEvent.index, order)])
df["TTE"] = timeToEvent.loc[order].values
df["TTE"] -= df["TTE"].min() - 1
df = pd.concat([dummy, df], axis=1)
# %%
import kaplanmeier as km
for i in range(1, groups.max()+1):
    print(i)
    kmf = km.fit(df[labels==1]["TTE"], df[labels==1]["Dead"].astype(int), groups[labels==1] == i)
    km.plot(kmf)
    plt.ylim(0,1)
    plt.show()
    plt.savefig(paths.outputDir + f"rnaseq/BRCA_clust/kaplan_{i}_vs_rest.pdf")
# %%
