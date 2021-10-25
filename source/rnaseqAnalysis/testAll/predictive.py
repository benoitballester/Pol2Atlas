# %%
import numpy as np
import pandas as pd
from settings import params, paths
from scipy.io import mmread
import os
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from kneed import KneeLocator
from settings import params, paths
from lib import normRNAseq
from scipy.special import expit
from sklearn.preprocessing import power_transform, PowerTransformer, QuantileTransformer
# %%
case = "TCGA-PRAD"

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

# %%
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
allCounts = np.concatenate(counts, axis=1)

# %%
scale = np.mean(allCounts, axis=0)
nz = np.sum(allCounts.T > 1, axis=0) > 2
countsNorm = allCounts.T / allReads[:, None]
countsNorm = countsNorm[:, nz]
countsNorm = countsNorm / np.min(countsNorm[countsNorm.nonzero()])

# %%
# Predictive model
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import catboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu, ttest_ind, ranksums
from statsmodels.stats.multitest import fdrcorrection
from sklearn.decomposition import PCA


labels = []
for a in annotation["Sample Type"].loc[order]:
    if a == "Solid Tissue Normal":
        labels.append(0)
    else:
        labels.append(1)
labels = np.array(labels)

# %%
predictions = np.zeros(len(labels), dtype=int)
for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(countsNorm, labels):
    pvals = mannwhitneyu(countsNorm[train][labels[train] == 0], countsNorm[train][labels[train] == 1])[1]
    pvals = np.nan_to_num(pvals, nan=1.0)
    meanFC = np.mean(countsNorm[train][labels[train] == 0], axis=0) / np.mean(countsNorm[train][labels[train] == 1], axis=0)
    kept = (fdrcorrection(pvals)[1] < 0.05) & (np.abs(np.log2(meanFC)) > 1)
    x_train = countsNorm[train][:, kept]
    # Fit classifier on train data
    model = catboost.CatBoostClassifier(iterations=100, rsm=np.sqrt(x_train.shape[1])/x_train.shape[1],
                                        class_weights=len(labels) / (2 * np.bincount(labels)))
    model.fit(x_train, labels[train])
    # Predict on test data
    x_test = countsNorm[test][:, kept]
    predictions[test] = model.predict(x_test)
print("Weighted accuracy :", balanced_accuracy_score(labels, predictions))
print("Recall :", recall_score(labels, predictions))
print("Precision :", precision_score(labels, predictions))
df = pd.DataFrame(confusion_matrix(labels, predictions))
df.columns = ["Normal Tissue True", "Tumor True"]
df.index = ["Normal Tissue predicted", "Tumor predicted"]

# %%
# DE Genes


qvals = findDE(countsNorm, labels)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
# consensuses = consensuses[nzPos][top]
topLocs = consensuses.iloc[(qvals < 0.05).nonzero()[0]]
topLocs.to_csv(paths.tempDir + f"topLocs{case}_DE.csv", sep="\t", header=None, index=None)
topLocs
# %%
from lib.utils import overlap_utils
import pyranges as pr

remap_catalog = pr.read_bed(paths.remapFile)
allPolII = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
topDEpr = topLocs
enrichments = overlap_utils.computeEnrichVsBg(remap_catalog, allPolII, topLocs)
enrichments[2].sort_values()[:50]
# %%
