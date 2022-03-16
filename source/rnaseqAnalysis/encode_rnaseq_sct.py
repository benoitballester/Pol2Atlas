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
import tensorflow as tf
import tensorflow_probability as tfp
class mult_layer(tf.keras.layers.Layer):
    def __init__(self, units=1):
        super(mult_layer, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        self.sfmults = self.add_weight(name='sf_mult',
                            shape=(input_shape[1],),
                            initializer='zeros',
                            trainable=True)
    def call(self, input):
        return tf.math.exp(self.sfmults) * input

def geomean(x, axis):
    return tf.math.exp(tf.math.reduce_mean(tf.math.log(x),axis=axis))


counts = tf.keras.layers.Input((510,))
sfs = tf.keras.layers.Input((510,))
# multiplie = tf.keras.layers.Dense(1, )
multiplier = mult_layer()(sfs)
multiplier = multiplier
sfModel = tf.keras.Model([counts, sfs], multiplier)
countsNorm = counts / multiplier
lambdaPoi = tfp.reduce_mean(counts, axis=1, keepdims=True)
countsModel = tf.keras.Model([counts, sfs], lambdaPoi)
dist = tfp.distributions.Poisson(lambdaPoi)
poiLogLike = 2*(dist.log_prob(lambdaPoi)-dist.log_prob(countsNorm))
l = tf.reduce_mean(tf.reduce_sum(poiLogLike, axis=1),50,axis=0)
model = tf.keras.Model([counts, sfs], l)
loss = lambda _, x:x
model.compile(tf.keras.optimizers.Adam(), loss=loss)
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=5, expMin=3)
counts = allCounts[:, nzCounts]
n_i = np.sum(counts, axis=1)
n_i = n_i / np.mean(n_i)
sfMat = np.tile(n_i[: ,None], counts.shape[1]).T
es = tf.keras.callbacks.EarlyStopping("loss", patience=5, restore_best_weights=True)
model.fit(x=(counts.T, sfMat), y=np.zeros(len(sfMat)), epochs=500, batch_size=128, callbacks=[es])
newSfs = sfModel.predict((counts.T[:1], sfMat[:1])).ravel()[:, None]
# %%
from statsmodels.genmod.families.family import Gaussian
def lrtTestPCA(matrix):
    model = PCA(200, whiten=True)
    decomp = model.fit_transform(matrix)
    mses=[np.sum([Gaussian().loglike(matrix[j, :], 0) for j, r in enumerate(matrix)])]
    for i in range(1, 200):
        recons = np.dot(decomp[:, :i], np.sqrt(model.explained_variance_[:i, None]) * model.components_[:i],)
        mse = np.sum([Gaussian().loglike(matrix[j, :], r) for j, r in enumerate(recons)])
        print(mses[-1]-mse)
        p = chi2.sf(-mses[-1]+mse, 2*np.prod(model.components_[0:1].shape))
        print(p)
        if p > 0.05:
            break
        mses.append(mse)
        print(i, mse)
    return PCA(i-1, whiten=True).fit_transform(matrix)
from sklearn.decomposition import PCA
from lib.jackstraw.permutationPA import permutationPA
from sklearn.preprocessing import StandardScaler
feat = StandardScaler().fit_transform(np.log(1+counts/newSfs))
# bestRank = permutationPA(feat, max_rank=min(100, len(rankQT)))
# model = PCA(42, whiten=True)
# decomp = lrtTestPCA(feat)
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5,
                     random_state=42, low_memory=False, metric="euclidean").fit_transform(feat)
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
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/umap_samples.pdf")
plt.show()

# %%
# %%
rowOrder = matrix_utils.threeStagesHC(feat, "correlation")
colOrder = matrix_utils.threeStagesHC(feat.T, "correlation")
# %%
plot_utils.plotHC(np.log(1+counts/n_i[:, None]).T, annotation.loc[order]["Annotation"], 
                  paths.polIIannotationPalette, rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/encode_rnaseq/encode_HM.pdf")
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
