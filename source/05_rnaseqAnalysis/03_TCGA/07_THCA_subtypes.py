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
from settings import params, paths
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import fdrcorrection

utils.createDir(paths.outputDir + "rnaseq/THCA")
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
case = "TCGA-THCA"
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
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
counts = allCounts
# %%
# Load molecular subtype info
subtypeAnnot = pd.read_csv(paths.tcgaSubtypes, sep=",", index_col="pan.samplesID")
subtypeAnnot.index = ["-".join(i.split("-")[:4]) for i in subtypeAnnot.index]
query = annotation.loc[order]["Sample ID"]
subtype = pd.Series(["None"]*len(query))
inSubtypeAnnot = np.isin(query, subtypeAnnot.index)
subtype[inSubtypeAnnot] = subtypeAnnot.loc[query[inSubtypeAnnot]]["Subtype_Selected"].values
sampleState = annotation.loc[order]["Sample Type"] == "Solid Tissue Normal"
subtype[sampleState.values] = "Solid Tissue Normal"

# %%
kept = (subtype != "None").values
annotation = annotation.loc[order][kept]
counts = counts[kept]
subtype = subtype[kept]
# %% 
catSubtype, eq = pd.factorize(subtype)
# %%
# Remove undected Pol II probes
nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
countsNz = counts[:, nzCounts]
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
decomp = rnaseqFuncs.permutationPA_PCA(residuals[:, hv], mincomp=2) 

# Plot PC 1 and 2
plt.figure(dpi=500)
plt.scatter(decomp[:, 0], decomp[:, 1], c=np.array(sns.color_palette())[catSubtype], s=0.5)
plt.show()
plt.close()
# %%
embedding = umap.UMAP(n_neighbors=30, min_dist=0.3,
                    random_state=42, low_memory=False, metric="correlation").fit_transform(decomp)
# %%
import seaborn as sns
plt.figure(figsize=(10,10), dpi=500)
plt.title(f"{case} samples")
patches = []
palette = np.array(sns.color_palette("Paired"))[[4,5,3,1,9,-1,-1]]
for i in np.unique(catSubtype):
    legend = Patch(color=palette[i], label=eq[i])
    patches.append(legend)
plt.legend(handles=patches)
plt.scatter(embedding[:,0], embedding[:, 1], c=palette[catSubtype], s=50, 
            linewidths=(0.0+1.0*(subtype.values=="Solid Tissue Normal")), edgecolors="k")
plt.savefig(paths.outputDir + "rnaseq/THCA/UMAP_subtypes.pdf")
plt.savefig(paths.outputDir + "rnaseq/THCA/UMAP_subtypes.png")
# %%
data = pd.DataFrame(embedding, columns=["x","y"])
data["Subtype"] = subtype.values
plt.figure(dpi=500)
g = sns.FacetGrid(data, col="Subtype", hue="Subtype", col_wrap=3, palette=palette)
g.map(sns.scatterplot, "x", "y")
plt.savefig(paths.outputDir + "rnaseq/THCA/UMAP_subtypes_facetgrid.pdf")
# %%
# Plot heatmap and dendrograms (hv)
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.residuals.T[hv], "correlation")
plot_utils.plotHC(residuals.T[hv], subtype, countsNorm.T[hv],  
                rowOrder=rowOrder, colOrder=colOrder, cmap="vlag", rescale="3SD")
plt.savefig(paths.outputDir + "rnaseq/THCA/HM_clusters.pdf")
# %%
import scipy.stats as ss
pctThreshold = 0.1
resAll = dict()
rankAll = dict()
lfcMin = 1.0
refGroup = subtype == "Solid Tissue Normal"
binVec = np.zeros((len(eq.drop(["Solid Tissue Normal"])), countModel.residuals.shape[1]), dtype="int")
j = 0
for i in eq.drop(["Solid Tissue Normal"]):
    print(i)
    res2 = ss.ttest_ind(countModel.residuals[subtype == i], countModel.residuals[refGroup], axis=0,
                        alternative="two-sided")
    sig, padj = fdrcorrection(res2[1])
    minpctM = np.mean(countsNz[subtype == i] > 0.5, axis=0) > max(pctThreshold, 1.5/(subtype == i).sum())
    minpctP = np.mean(countsNz[refGroup] > 0.5, axis=0) > max(pctThreshold, 1.5/(refGroup).sum())
    minpct = minpctM | minpctP
    fc = np.mean(countModel.normed[subtype == i], axis=0) / (1/refGroup.sum()+np.mean(countModel.normed[refGroup], axis=0))
    lfc = np.abs(np.log2(fc)) > lfcMin
    print(sig.sum())
    sig = sig & lfc & minpct
    print(sig.sum())
    res = pd.DataFrame(res2[::-1], columns=consensuses.index[nzCounts], index=["pval", "t-stat"]).T
    res["DE"] = sig.astype(int)
    ranked = np.lexsort((-np.abs(res["t-stat"].values), res["pval"].values))
    res = res.iloc[ranked]
    resAll[i] = res
    rankAll[i] = ranked
    res.to_csv(paths.outputDir + f"rnaseq/THCA/res_{i}.csv")
    test = consensuses[nzCounts]
    test["Score"] = res["t-stat"]
    test = test.iloc[ranked][sig[ranked]]
    test.to_csv(paths.outputDir + f"rnaseq/THCA/bed_{i}", header=None, sep="\t", index=None)
    delta = np.mean(countModel.residuals[subtype == i], axis=0) - np.mean(countModel.residuals[refGroup], axis=0)
    allWithScore = consensuses.copy()[["Chromosome", "Start", "End", "Name"]]
    allWithScore["logFDR"] = 0.0
    allWithScore["logFDR"][nzCounts] = -np.log10(padj)
    allWithScore["DeltaRes"] = 0.0
    allWithScore["DeltaRes"][nzCounts] = delta
    allWithScore["LFC"] = 0.0
    allWithScore["LFC"][nzCounts] = np.log2(fc)
    allWithScore.to_csv(paths.outputDir + "rnaseq/THCA/" + f"allWithStats_{i}.bed", sep="\t", index=None)
    binVec[j, sig] = 1 
    j += 1
    if len(test) == 0:
        continue
    """pvals = enricher.findEnriched(test, background=consensuses)
    enricher.plotEnrichs(pvals)
    enricher.clusterTreemap(pvals, score="-log10(pval)", 
                            output=paths.outputDir + f"rnaseq/THCA/great_{i}.pdf")
    pvals = enricherMF.findEnriched(test, background=consensuses)
    enricherMF.plotEnrichs(pvals)
    enricherMF.clusterTreemap(pvals, score="-log10(pval)", 
                            output=paths.outputDir + f"rnaseq/THCA/greatMF_{i}.pdf")
    """
# %%
# Unique marker per subtype
plt.hist(np.sum(binVec, axis=0), np.arange(5))
# %%
plt.hist(np.sum(binVec, axis=0), np.arange(5))
j = 0
for i in eq.drop(["Solid Tissue Normal"]):
    print(i)
    subtypeUnique = (np.sum(binVec, axis=0).astype(int) == 1) & (binVec[j] == 1)
    print(subtypeUnique.sum())
    test = consensuses[nzCounts]
    test["Score"] = resAll[i]["t-stat"]
    test = test.iloc[rankAll[i]][subtypeUnique[rankAll[i]]]
    test.to_csv(paths.outputDir + f"rnaseq/THCA/bed_uniqueDE_{i}", header=None, sep="\t", index=None)
    """pvals = enricher.findEnriched(test, background=consensuses)
    enricher.plotEnrichs(pvals)
    enricher.clusterTreemap(pvals, score="-log10(pval)", 
                            output=paths.outputDir + f"rnaseq/THCA/unique_{i}.pdf")
    pvals = enricherMF.findEnriched(test, background=consensuses)
    enricherMF.plotEnrichs(pvals)
    enricherMF.clusterTreemap(pvals, score="-log10(pval)", 
                            output=paths.outputDir + f"rnaseq/THCA/uniqueMF_{i}.pdf") """
    j += 1

# %%
with open(paths.tempDir + "end0507.txt", "w") as f:
    f.write("1")
