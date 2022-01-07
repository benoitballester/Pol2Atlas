# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib import rnaseqFuncs
from lib.utils import plot_utils, matrix_utils
from matplotlib.patches import Patch
from scipy.stats import rankdata, chi2, mannwhitneyu, ttest_ind
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,balanced_accuracy_score
import catboost

# %%
try:
    os.mkdir(paths.outputDir + "rnaseq/TumorVsNormal/")
except FileExistsError:
    pass
perCancerDE = {}
# %%
case = "TCGA-ESCA"
try:
    os.mkdir(paths.outputDir + "rnaseq/TumorVsNormal/" + case)
except FileExistsError:
    pass
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
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
# %%
# Remove undected Pol II probes
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=3)
counts = allCounts[:, nzCounts]
# Convert reads to ranks
ranks = rankdata(counts, "min", axis=1)
# Rescale ranks to unit Variance for numerical stability (assuming uniform distribution for ranks)
rgs = (ranks / ranks.shape[1] - 0.5) * np.sqrt(12)
# %%
# Find DE genes
countsTumor = ranks[labels == 1]
countsNormal = ranks[labels == 0]
stat, pvals = mannwhitneyu(countsNormal, countsTumor)
pvals = np.nan_to_num(pvals, nan=1.0)
qvals = fdrcorrection(pvals)[1]
perCancerDE[case] = np.zeros(allCounts.shape[1])
perCancerDE[case][nzCounts] = qvals < 0.05
# %%
# Plot DE signal
orderCol = np.argsort(labels)
meanNormal = np.mean(ranks[labels == 0], axis=0)
meanTumor = np.mean(ranks[labels == 1], axis=0)
diff = meanTumor - meanNormal
orderRow = np.argsort(diff)
plot_utils.plotDeHM(ranks[orderCol][:, orderRow], labels[orderCol], (qvals < 0.05).astype(float)[orderRow])
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/" + case + "/DE_plot.pdf")
plt.show()
plt.close()
# %%
# UMAP plot
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5,
                     random_state=42, low_memory=False, metric="correlation").fit_transform(rgs[:, qvals < 0.05])
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
# %%
# Predictive model on DE Pol II
predictions = np.zeros(len(labels), dtype=int)
for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(rgs[:, qvals < 0.05], labels):
    # Fit power transform on train data only
    x_train = rgs[train][:, qvals < 0.05]
    # Fit classifier on scaled train data
    model = catboost.CatBoostClassifier(iterations=100, rsm=np.sqrt(x_train.shape[1])/x_train.shape[1],
                                        class_weights=len(labels) / (2 * np.bincount(labels)), random_state=42)
    model.fit(x_train, labels[train])
    # Scale and predict on test data
    x_test = rgs[test][:, qvals < 0.05]
    predictions[test] = model.predict(x_test)
with open(paths.tempDir + f"classifier_{case}.txt", "w") as f:
    print("Weighted accuracy :", balanced_accuracy_score(labels, predictions), file=f)
    wAccs[case] = balanced_accuracy_score(labels, predictions)
    print("Recall :", recall_score(labels, predictions), file=f)
    print("Precision :", precision_score(labels, predictions), file=f)
    df = pd.DataFrame(confusion_matrix(labels, predictions))
    df.columns = ["Normal Tissue True", "Tumor True"]
    df.index = ["Normal Tissue predicted", "Tumor predicted"]
    print(df, file=f)
print("Weighted accuracy :", balanced_accuracy_score(labels, predictions))
print("Recall :", recall_score(labels, predictions))
print("Precision :", precision_score(labels, predictions))
df = pd.DataFrame(confusion_matrix(labels, predictions))
df.columns = ["Normal Tissue True", "Tumor True"]
df.index = ["Normal Tissue predicted", "Tumor predicted"]
# %%
