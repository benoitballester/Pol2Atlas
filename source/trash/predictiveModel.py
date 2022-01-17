# %%
import numpy as np
import pandas as pd
from settings import params, paths
from scipy.io import mmread
import matplotlib.pyplot as plt

counts = np.asarray(mmread(paths.outputDir + "rnaseq/" + "counts.mtx").todense())
annotation = pd.read_csv(paths.outputDir + "rnaseq/annotation.tsv", sep="\t")

# %%
from scipy.stats.mstats import gmean
'''
countsUQ = np.zeros_like(counts, dtype="float")
for i in range(len(counts)):
    countsUQ[i] = counts[i] / np.percentile(counts[i][counts[i].nonzero()],75)
countsUQ /= np.min(countsUQ[countsUQ.nonzero()])
'''
# RLE normalization
m = gmean(counts, axis=0)
c1 = np.where(m > 1e-10, counts / m, np.nan)
scales = np.nanmedian(c1, axis=1)[:, None]
countsRLE = counts / scales

# Remove Pol II with very low expression
nzPos = np.sum(countsRLE > 5, axis=0) > 2
countsRLE = countsRLE[:, nzPos]
# Outlier removal (extreme max read counts > 2 * non-zero 99th percentile)
perc99 = np.percentile(countsRLE[countsRLE.nonzero()].ravel(), 99)

nonOutliers = np.max(countsRLE, axis=0) < perc99*2
countsRLE = countsRLE[:,  nonOutliers]
plt.figure()
plt.hist(countsRLE.ravel(), 20)
plt.yscale("log")
plt.xlabel("FPKM-UQ counts")
plt.show()
# %%
# Log transform, scale to unit variance and zero mean
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
countsScaled = np.log10(1+countsRLE)
countsScaled = scaler.fit_transform(countsScaled)
plt.figure()
plt.hist(countsScaled.ravel(), 20)
plt.yscale("log")
plt.xlabel("Z-scores")
plt.show()
# %%
# Outliers :/
plt.figure(dpi=500)
plt.boxplot(countsScaled[:100].T,showfliers=False)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()
# %%
# Remove experiments with abnormal IQR
IQRs = np.percentile(countsScaled, 75, axis=1)-np.percentile(countsScaled, 25, axis=1)
IQR95 = np.percentile(IQRs, 95)
nonOutliers = IQRs < IQR95*2
countsScaled = countsScaled[nonOutliers]
plt.figure(dpi=500)
plt.boxplot(countsScaled[:100].T,showfliers=False)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()

# %%
labels = []
for a in annotation["State"][nonOutliers]:
    if a == "Solid Tissue Normal":
        labels.append(0)
    else:
        labels.append(1)
labels = np.array(labels)
# %%
# PCA, determine optimal number of K using the elbow method
from sklearn.decomposition import PCA
from kneed import KneeLocator
pc = PCA(100, whiten=True)
decomp = pc.fit_transform(countsScaled)
kneedl = KneeLocator(np.arange(100), pc.explained_variance_, direction="decreasing", curve="convex")
bestR = kneedl.knee
plt.figure(dpi=300)
kneedl.plot_knee()
plt.xlabel("Principal component")
plt.ylabel("Explained variance")
plt.show()
decomp = decomp[:, :bestR]
# %%
# Tumor prediction (Catboost model)
from lib.utils import matrix_utils
k = 1
acc = matrix_utils.looKnnCV(decomp, labels, "euclidean", k)
print(f"{k}-NN leave-one-out cross validation accuracy : {acc}")
# %% 
# Tumor prediction (Catboost model)
from sklearn.model_selection import train_test_split, StratifiedKFold
import catboost
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,balanced_accuracy_score

predictions = np.zeros(len(labels), dtype=int)
for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(decomp, labels):
    model = catboost.CatBoostClassifier(class_weights=len(labels) / (2 * np.bincount(labels)))
    model.fit(decomp[train], labels[train], verbose=False)
    predictions[test] = model.predict(decomp[test])
print("Weighted accuracy :", balanced_accuracy_score(labels, predictions))
print("Recall :", recall_score(labels, predictions))
print("Precision :", precision_score(labels, predictions))
df = pd.DataFrame(confusion_matrix(labels, predictions))
df.columns = ["Normal Tissue True", "Tumor True"]
df.index = ["Normal Tissue predicted", "Tumor predicted"]
print(df)

# %%
import umap
embedding = umap.UMAP(metric="euclidean").fit_transform(decomp)

# %%
import matplotlib.pyplot as plt
from lib.utils.plot_utils import plotUmap, getPalette
plt.figure(dpi=500)
palette, colors = getPalette(labels)
plotUmap(embedding, colors)
plt.show()
# %%
