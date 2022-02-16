# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib import rnaseqFuncs, normRNAseq
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
palette = pd.read_csv(paths.polIIannotationPalette, sep="\t")
palette = dict([(d["Annotation"], (d["r"],d["g"],d["b"])) for r,d in palette.iterrows()])
fpkmExpr = np.sum(allCounts/allReads[:, None], axis=1)*100
df = pd.DataFrame(data=np.concatenate([fpkmExpr[:, None], annotation.loc[order]["Annotation"].ravel()[:, None]], axis=1), columns=["Percentage of mapped reads", "Annotation"])
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, showfliers=False)
sns.stripplot(data=df, x="Percentage of mapped reads", y="Annotation", palette=palette, dodge=True, 
                edgecolor="black", jitter=1/3, alpha=1.0, s=2, linewidth=0.1)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/pctmapped_per_annot.pdf", bbox_inches="tight")
# %%
# Remove undected Pol II probes
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scran = importr("scran")

counts = allCounts
sf = np.sum(counts, axis=1)
sf = sf / np.min(sf)
countsNorm = counts/sf[:, None]
nzCounts = rnaseqFuncs.filterDetectableGenes(countsNorm, readMin=1, expMin=3)
countsNorm = countsNorm[:, nzCounts]
logCounts = np.log(1+countsNorm)
# %%
# Feature selection based on rank variability
selected = rnaseqFuncs.variableSelection(rankdata(countsNorm, "min", axis=1), plot=True)
# Apply normal quantile transform per gene
rankQT = rnaseqFuncs.quantileTransform(countsNorm)
# %%
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod.families import NegativeBinomial as nbd
from scipy.stats import nbinom
import tensorflow_probability as tfp
residuals = []
for i in range(1000):
    try:
        df = countsNorm[:, nzCounts][:, selected][:, i]
        model = NegativeBinomial(df, np.ones_like(df)).fit()
        dev = nbd(link=None, alpha=model.params[1]).deviance(df, model.predict(np.ones_like(df)))
        residuals.append(nbd(link=None, alpha=model.params[1]).resid_dev(df, model.predict(np.ones_like(df))))
    except:
        continue
print(dev)
print(dev_res)
print(model.summary())
# %%
from sklearn.decomposition import PCA
factors = PCA(30, whiten=True).fit_transform(np.array(residuals).T)
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5,
                     random_state=42, low_memory=False, metric="correlation").fit_transform(factors)
plt.figure(figsize=(10,10), dpi=500)
palette, colors = plot_utils.getPalette(ann)
plot_utils.plotUmap(embedding, colors)
patches = []
for i in np.unique(ann):
    legend = Patch(color=palette[i], label=eq[i])
    patches.append(legend)
plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
# plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/umap_samples.pdf")
plt.show()

# %%
