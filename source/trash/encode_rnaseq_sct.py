# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib import normRNAseq, rnaseqFuncs
from lib.utils import plot_utils, matrix_utils
from matplotlib.patches import Patch
from scipy.stats import rankdata, chi2
from scipy.stats import chi2
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection
from scipy.spatial.distance import dice
import matplotlib as mpl
import fastcluster
import sklearn.metrics as metrics

countDir = "/scratch/pdelangen/projet_these/outputPol2/rnaseq/encode_counts/"
try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/")
except FileExistsError:
    pass
# %%
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/encode_total_rnaseq_annot_0.tsv", 
                        sep="\t", index_col=0)
dlFiles = os.listdir(countDir + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
for f in dlFiles:
    try:
        id = f.split(".")[0]
        # countsBG.append(pd.read_csv(paths.countDirectory + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except:
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
ann, eq = pd.factorize(annotation.loc[order]["Annotation"])
# %% 
# Plot FPKM expr per annotation
palette = pd.read_csv(paths.polIIannotationPalette, sep=",")
palette = dict([(d["Annotation"], (d["r"],d["g"],d["b"])) for r,d in palette.iterrows()])
fpkmExpr = np.sum(allCounts/allReads[:, None], axis=1)*100
df = pd.DataFrame(data=np.concatenate([fpkmExpr[:, None], annotation.loc[order]["Annotation"].ravel()[:, None]], axis=1), columns=["Percentage of mapped reads", "Annotation"])
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, showfliers=False)
sns.stripplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, dodge=True, 
                edgecolor="black", jitter=1/3, alpha=1.0, s=2, linewidth=0.1)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/pctmapped_per_annot.pdf", bbox_inches="tight")
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
from lib.AE import DenoiserAE
model = DenoiserAE("NB")
denoised = model.fit_transform(counts.T).T
# %%
from statsmodels.genmod.families.family import Gamma, NegativeBinomial, Poisson
from statsmodels.genmod.families.links import Log
import statsmodels.api as sm
func = NegativeBinomial(Log(), 1.0)
residuals = np.zeros_like(denoised)
denoisedNormed = denoised / sf[:, None]
means = np.mean(denoisedNormed, axis=0)
for i in range(denoised.shape[1]):
    residuals[:, i] = func.resid_anscombe(denoisedNormed[:, i], means[i])
    if i % 200 == 0:
        print(i)
# %%
from scipy.stats import chi2
deviances = np.sum(residuals**2, axis=0)
pDev = chi2(509).sf(deviances)
hv = fdrcorrection(pDev)[0]
lv = fdrcorrection(1-pDev)[0]

# %%
from sklearn.decomposition import PCA
from lib.jackstraw.permutationPA import permutationPA
from sklearn.preprocessing import StandardScaler

feat = residuals[:, hv]
bestRank = permutationPA(feat, max_rank=min(200, np.min(feat.shape))-1)
modelPCA = PCA(bestRank[0], whiten=True, svd_solver="arpack", random_state=42)
decomp = modelPCA.fit_transform(feat)
matrix_utils.looKnnCV(decomp, ann, "correlation",1)
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=42, low_memory=False, metric="correlation").fit_transform(decomp)
plt.figure(figsize=(10,10), dpi=500)
annot, palette, colors = plot_utils.applyPalette(annotation.loc[order]["Annotation"],
                                                np.unique(annotation.loc[order]["Annotation"]),
                                                 paths.polIIannotationPalette, ret_labels=True)
plot_utils.plotUmap(embedding, colors)
patches = []
for i, a in enumerate(annot):
    legend = Patch(color=palette[i], label=a)
    patches.append(legend)
plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
# plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/umap_samples.pdf")
plt.show()
plt.close()

# %%
# %%
rowOrder, rowLinakge = matrix_utils.twoStagesHClinkage(decomp, "correlation")
colOrder = matrix_utils.threeStagesHC(feat.T, "correlation")
# %%
plot_utils.plotHC(np.log10(1+counts[:, (hv & outliers)]/sf[:, None]).T, annotation.loc[order]["Annotation"], (counts[:, (hv & outliers)]/sf[:, None]).T,
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
# plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM.pdf")
# %%
# Plot FPKM expr per annotation
palette = pd.read_csv(paths.polIIannotationPalette, sep=",")
palette = dict([(d["Annotation"], (d["r"],d["g"],d["b"])) for r,d in palette.iterrows()])
fpkmExpr = np.sum(allCounts[:, nzCounts]/newSfs, axis=1)*100

df = pd.DataFrame(data=np.concatenate([fpkmExpr[:, None], annotation.loc[order]["Annotation"].ravel()[:, None]], axis=1), columns=["Percentage of mapped reads", "Annotation"])
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, showfliers=False)
sns.stripplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, dodge=True, 
                edgecolor="black", jitter=1/3, alpha=1.0, s=2, linewidth=0.1)
# %%
# Remove undected Pol II probes
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scran = importr("scran")

counts = allCounts
sfScran = scran.calculateSumFactors(counts.T, scaling=allReads[:, None])
# %%
# Plot FPKM expr per annotation
palette = pd.read_csv(paths.polIIannotationPalette, sep=",")
palette = dict([(d["Annotation"], (d["r"],d["g"],d["b"])) for r,d in palette.iterrows()])
fpkmExpr = np.sum(allCounts/np.array(sfScran)[:, None], axis=1)*100
df = pd.DataFrame(data=np.concatenate([fpkmExpr[:, None], annotation.loc[order]["Annotation"].ravel()[:, None]], axis=1), columns=["Percentage of mapped reads", "Annotation"])
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, showfliers=False)
sns.stripplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, dodge=True, 
                edgecolor="black", jitter=1/3, alpha=1.0, s=2, linewidth=0.1)
# %%