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
from matplotlib.ticker import FormatStrFormatter
# %%
'''
from lib.pyGREATglm import pyGREAT
import pyranges as pr
enricher = pyGREAT(oboFile=paths.GOfolder + "/go_eq.obo", geneFile=paths.gencode, 
                   geneGoFile=paths.GOfolder + "/goa_human.gaf")
'''
# %%
allAnnots = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
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
# Compute DE, UMAP, and predictive model CV per cancer
case = "TCGA-ESCA"
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
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
# %%
geneTable = pd.read_hdf("/scratch/pdelangen/projet_these/data_clean/geneCounts.hd5")
#%%
geneTableAnnot = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotationCounts.tsv", index_col="Sample ID", sep="\t")
used = geneTableAnnot.loc[annotation["Sample ID"]]["File ID"]
usedTable = geneTable[used].astype("float").iloc[:-5].T
# %%
sf = normRNAseq.deseqNorm(usedTable.values)
geneTableNorm = usedTable/sf
# %%
from sklearn.preprocessing import PowerTransformer
counts = allCounts
countsNorm = counts/sf
# Remove undected Pol II probes
nzCounts = rnaseqFuncs.filterDetectableGenes(countsNorm, readMin=1, 
                                            expMin=3)
studiedConsensusesCase[case] = nzCounts.nonzero()[0]
countsNorm = countsNorm[:, nzCounts]
rgs = countsNorm
# Find DE genes
countsTumor = rgs[labels == 1]
countsNormal = rgs[labels == 0]
stats, pvals = mannwhitneyu(countsNormal, countsTumor)
pvals = np.nan_to_num(pvals, nan=1.0)
qvals = fdrcorrection(pvals)[1]
allDE = consensuses[nzCounts][qvals < 0.05]
allDE[4] = -np.log10(qvals[qvals < 0.05])