# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib import rnaseqFuncs
from lib.utils import plot_utils, matrix_utils
from matplotlib.patches import Patch
from scipy.stats import rankdata, chi2
from scipy.stats import chi2
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scran = importr("scran")
# %%
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
dlFiles = os.listdir(paths.countDirectory + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
for f in np.array(dlFiles):
    try:
        id = f.split(".")[0]
        # countsBG.append(pd.read_csv(paths.countDirectory + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(paths.countDirectory + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(paths.countDirectory + "500centroid/" + f, header=None, skiprows=2).values)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except:
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
# bgCounts = np.concatenate(countsBG, axis=1).T
# %%
# Keep tumoral samples
kept = np.isin(order, annotation.index)
allCounts = allCounts[kept]
# bgCounts = bgCounts[:, kept]
allReads = allReads[kept]
annotation = annotation.loc[np.array(order)[kept]]
tumor = np.logical_not(annotation["Sample Type"] == "Solid Tissue Normal")
# %%
# Remove undected Pol II probes
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
countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf)
pDev, outliers = countModel.hv_selection()
hv = fdrcorrection(pDev)[0]
lv = fdrcorrection(1-pDev)[0]

# %%
from sklearn.decomposition import PCA
from lib.jackstraw.permutationPA import permutationPA_PCA
from sklearn.preprocessing import StandardScaler

feat = countModel.anscombeResiduals[:, hv & outliers]
decomp = permutationPA_PCA(feat, 10, max_rank=min(500, len(feat)))
print(decomp.shape)
# %%
import umap
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(decomp)
#%%
import catboost
from sklearn.model_selection import train_test_split, StratifiedKFold
project_id = annotation["project_id"]
# cancerType, eq = pd.factorize(tcgaProjects["Origin"].loc[project_id])
cancerType, eq = pd.factorize(annotation["project_id"], sort=True)
pred = np.zeros(len(cancerType), dtype=int)
for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(decomp, cancerType):
    # Fit power transform on train data only
    x_train = decomp[train]
    # Fit classifier on scaled train data
    model = catboost.CatBoostClassifier(class_weights=len(cancerType) / (len(np.unique(cancerType)) * np.bincount(cancerType)), random_state=42)
    model.fit(x_train, cancerType[train], silent=True)
    # Scale and predict on test data
    x_test = decomp[test]
    pred[test] = model.predict(x_test).ravel()
# %%
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,balanced_accuracy_score
acc = balanced_accuracy_score(cancerType, pred)
# %%
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch


plt.figure(dpi=500)
palette, colors = getPalette(cancerType)
# allReadsScaled = (allReads - allReads.min()) / (allReads.max()-allReads.min())
# col = sns.color_palette("rocket_r", as_cmap=True)(allReadsScaled)
scale = min(10.0,100/np.sqrt(len(embedding)))
plt.scatter(embedding[:, 0], embedding[:, 1], s=scale*(3.0-2.0*tumor),
            linewidths=(1.0-tumor)*0.5, edgecolors=(0.1,0.1,0.1),c=colors)
xScale = plt.xlim()[1] - plt.xlim()[0]
yScale = plt.ylim()[1] - plt.ylim()[0]
# plt.gca().set_aspect(xScale/yScale)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/global/umap_all_tumors_plus_normal.png")
plt.show()
plt.figure(dpi=500)
plt.axis('off')
patches = []
for i in np.unique(cancerType):
    legend = Patch(color=palette[i], label=eq[i][5:])
    patches.append(legend)
plt.legend(handles=patches)
plt.savefig(paths.outputDir + "rnaseq/global/umap_all_tumors_lgd.png", bbox_inches="tight")
plt.title(f"Balanced accuracy on {len(np.unique(cancerType))} cancers : {acc}")
plt.show()

# %%

clustsPol2 = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt",dtype=int)[nzCounts]
nClusts = np.max(clustsPol2)+1
nAnnots = len(eq)
zScores = np.zeros((nClusts, nAnnots))
avg1Read = np.mean(countModel.anscombeResiduals, axis=0) > 1
filteredMat = np.log(1+countModel.anscombeResiduals)[:, avg1Read]
for i in range(nAnnots):
    hasAnnot = cancerType == i
    sd = np.std(np.percentile(filteredMat[hasAnnot], 95, axis=0))
    expected = np.mean(np.percentile(filteredMat[hasAnnot], 95, axis=0))
    for j in range(nClusts):
        inClust = clustsPol2[avg1Read] == j
        notInClust = np.logical_not(clustsPol2 == j)
        observed = np.mean(np.percentile(filteredMat[hasAnnot][:, inClust], 95, axis=0))
        zScores[j, i] = (observed-expected)/sd
# %%
import matplotlib as mpl
rowOrder, colOrder = matrix_utils.HcOrder(zScores)
rowOrder = np.loadtxt(paths.outputDir + "clusterBarplotOrder.txt").astype(int)
zClip = np.clip(zScores,0.0,10.0)
zNorm = np.clip(zClip / np.percentile(zClip, 95),0.0,1.0)
plt.figure(dpi=300)
sns.heatmap(zNorm[rowOrder].T[colOrder], cmap="vlag", linewidths=0.1, linecolor='black', cbar=False)
plt.gca().set_aspect(2.0)
plt.yticks(np.arange(len(eq))+0.5, eq[colOrder], rotation=0)
plt.xticks([],[])
plt.xlabel(f"{len(zNorm)} Pol II clusters")
plt.savefig(paths.outputDir + "rnaseq/global/signalPerClustPerAnnot.pdf", bbox_inches="tight")
plt.show()
plt.figure(figsize=(6, 1), dpi=300)
norm = mpl.colors.Normalize(vmin=0, vmax=np.percentile(zClip, 95))
cb = mpl.colorbar.ColorbarBase(plt.gca(), sns.color_palette("vlag", as_cmap=True), norm, orientation='horizontal')
cb.set_label("95th percentile Z-score")
plt.tight_layout()
plt.savefig(paths.outputDir + "rnaseq/global/signalPerClustPerAnnot_colorbar.pdf")
plt.show()
# %%
