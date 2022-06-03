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
from scipy.spatial.distance import dice
import matplotlib as mpl
import fastcluster
import sklearn.metrics as metrics
import scipy.stats as ss

countDir = "/scratch/pdelangen/projet_these/outputPol2/rnaseq/gtex_counts/"
try:
    os.mkdir(paths.outputDir + "rnaseq/gtex_rnaseq/")
except FileExistsError:
    pass
# %%
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/GTex/tsvs/sample.tsv", 
                        sep="\t", index_col="specimen_id")
colors = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/GTex/colors.txt", 
                        sep="\t", index_col="tissue_site_detail")
dlFiles = os.listdir(countDir + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
allStatus = []
for f in dlFiles:
    try:
        id = ".".join(f.split(".")[:-3])
        # countsBG.append(pd.read_csv(paths.countDirectory + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                                header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values)
        allStatus.append(status)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(f.split(".")[0])
    except:
        print(f, "missing")
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
ann, eq = pd.factorize(annotation.loc[order]["tissue_type"])
# %% 
# Plot FPKM expr per annotation
palette = pd.read_csv(paths.polIIannotationPalette, sep=",")
palette = dict([(d["Annotation"], (d["r"],d["g"],d["b"])) for r,d in palette.iterrows()])
fpkmExpr = np.sum(allCounts/allReads[:, None], axis=1)*100
df = pd.DataFrame(data=np.concatenate([fpkmExpr[:, None], annotation.loc[order]["tissue_type"].ravel()[:, None]], axis=1), columns=["Percentage of mapped reads", "Annotation"])
plt.figure(figsize=(6,4), dpi=500)
sns.boxplot(data=df, x="Percentage of mapped reads", y="Annotation", showfliers=False)
sns.stripplot(data=df, x="Percentage of mapped reads", y="Annotation", dodge=True, 
                edgecolor="black", jitter=1/3, alpha=1.0, s=2, linewidth=0.1)
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/pctmapped_per_annot.pdf", bbox_inches="tight")
# %%
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=2)
counts = allCounts[:, nzCounts]

# %%
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
scran = importr("scran")
base = importr("base")
detected = [np.sum(counts >= i, axis=0) for i in range(20)][::-1]
topMeans = np.lexsort(detected)[::-1][:int(counts.shape[1]*0.05+1)]
with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
    sf = scran.calculateSumFactors(counts.T[topMeans])

# %%
countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf)

hv = countModel.hv

# %%
feat = countModel.residuals[:, hv]
decomp, model = rnaseqFuncs.permutationPA_PCA(feat, returnModel=True)
matrix_utils.looKnnCV(decomp, ann, "correlation", 5)
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(decomp)
# %%
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
df["Organ"] = annotation.loc[order]["tissue_type"].values
df["Organ detailled"] = annotation.loc[order]["tissue_type_detail"].values
df = df.sample(frac=1)
colormap = dict(zip(colors.index, colors["color_hex"])) 
fig = px.scatter(df, x="x", y="y", color="Organ detailled", color_discrete_map=colormap,
                hover_data=['Organ detailled'], width=1200, height=800)
fig.update_traces(marker=dict(size=3*np.sqrt(len(df)/7500)))
fig.show()
fig.write_image(paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples.pdf")
fig.write_html(paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples.pdf" + ".html")
# %%
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
df["Total reads"] = np.sum(counts, axis=1)
fig = px.scatter(df, x="x", y="y", color="Total reads", width=1200, height=800)
fig.show()

# %%
plt.figure(figsize=(10,10), dpi=500)
palette, colors = plot_utils.getPalette(ann)
plot_utils.plotUmap(embedding, colors)
patches = []
for i, a in enumerate(eq):
    legend = Patch(color=palette[i], label=a)
    patches.append(legend)
plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples.pdf")
plt.show()
plt.close()
# %%
try:
    os.mkdir(paths.outputDir + "rnaseq/gtex_rnaseq/DE/")
except FileExistsError:
    pass
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
from lib.pyGREATglm import pyGREAT as pyGREATglm
enricherglm = pyGREATglm("/scratch/pdelangen/projet_these/data_clean/GO_files/hsapiens.GO:BP.name.gmt",
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
# %%
for i in np.unique(ann):
    print(eq[i])
    labels = (ann == i).astype(int)
    res2 = rnaseqFuncs.mannWhitneyDE(countModel.residuals, sf, labels, order)
    sig = fdrcorrection(res2[0])[0] & (res2[1] > 0)
    res = pd.DataFrame(res2, columns=consensuses.index[nzCounts], index=["pval", "residual diff"]).T
    res["Upreg"] = sig.astype(int)
    res.to_csv(paths.outputDir + f"rnaseq/gtex_rnaseq/DE/res_{eq[i]}.csv")
    test = consensuses[nzCounts][sig]
    test.to_csv(paths.outputDir + f"rnaseq/gtex_rnaseq/DE/bed_{eq[i]}", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
    pvals = enricherglm.findEnriched(test, background=consensuses)
    enricherglm.plotEnrichs(pvals)
    enricherglm.clusterTreemap(pvals, score="-log10(pval)", 
                                output=paths.outputDir + f"rnaseq/gtex_rnaseq/DE/great_{eq[i]}.pdf")
# %%
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(feat.T, "correlation")
# %%
# Plot dendrograms
from scipy.cluster import hierarchy
plt.figure(dpi=500)
hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/gtex_HM_hvg_col_dendrogram.pdf")
plt.show()
plt.close()
# Plot dendrograms
plt.figure(dpi=500)
hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1, orientation="left")
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/gtex_HM_row_dendrogram.pdf")
plt.show()
plt.close()
# %%
clippedSQ= np.log(1+countModel.normed)
plot_utils.plotHC(clippedSQ.T[hv], annotation.loc[order]["tissue_type"], (countModel.normed).T[hv & nonOutliers],  
                  rowOrder=rowOrder, colOrder=colOrder)
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/gtex_HM_hvg.pdf", bbox_inches="tight")
# %%
colOrderAll, colLinkAll = matrix_utils.threeStagesHClinkage(countModel.anscombeResiduals.T, "correlation")
# Plot dendrograms
plt.figure(dpi=500)
hierarchy.dendrogram(colLinkAll, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/gtex_HM_col_dendrogram.pdf")
plt.show()
clippedSQ= np.log(1+countModel.normed)
plt.figure(dpi=500)
plot_utils.plotHC(clippedSQ.T, annotation.loc[order]["tissue_type"], (countModel.normed).T,  
                  rowOrder=rowOrder, colOrder=colOrderAll)
plt.savefig(paths.outputDir + "rnaseq/gtex_rnaseq/gtex_HM.pdf", bbox_inches="tight")
