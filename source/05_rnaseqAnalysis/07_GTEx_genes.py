# %%
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("./")
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
# %%
import pandas as pd
df = pd.read_csv("/shared/projects/pol2_chipseq/test/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct", sep="\t", skiprows=2).T
df 
# %%
counts = df.iloc[2:].values.astype("int32")
order = np.array(df.index[2:])
# %%
annotation = pd.read_csv(paths.gtexData + "/tsvs/sample.tsv", 
                        sep="\t", index_col="specimen_id")
colors = pd.read_csv(paths.gtexData + "colors.txt", 
                        sep="\t", index_col="tissue_site_detail")
order = order[np.isin(order, annotation.index)]
conv = pd.read_csv(paths.tissueToSimplified, sep="\t", index_col="Tissue")
annotation.loc[order, "tissue_type"] = conv.loc[annotation.loc[order]["tissue_type"].values]["Simplified"].values
annTxt = annotation.loc[order]["tissue_type"]
ann, eq = pd.factorize(annTxt)
# %%
counts = counts[np.isin(df.index[2:], order)]
nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
counts = counts[:, nzCounts]
# %%
sf = rnaseqFuncs.scranNorm(counts)
countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf)
hv = countModel.hv
# %%
feat = countModel.residuals[:, hv]
decomp = rnaseqFuncs.permutationPA_PCA(feat, 1, max_rank=1000, returnModel=False)
fullAnn = annotation.loc[order]["tissue_type_detail"].values
annFull, eqFull = pd.factorize(fullAnn)
acc = matrix_utils.looKnnCV(decomp, annFull, "correlation", 5)
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(feat)
# %%
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
df["Organ"] = annotation.loc[order]["tissue_type"].values
df["Organ detailled"] = annotation.loc[order]["tissue_type_detail"].values
df = df.sample(frac=1)
colormap = dict(zip(colors.index, colors["color_hex"])) 
fig = px.scatter(df, x="x", y="y", color="Organ detailled", color_discrete_map=colormap,
                hover_data=['Organ detailled'], width=1200, height=800, title=f"54 Tissue KNN classification balanced accuracy : {acc}")
fig.update_traces(marker=dict(size=3*np.sqrt(len(df)/7500)))
fig.show()
fig.write_image(paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples_gene_centric.pdf")
fig.write_html(paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples_gene_centric.pdf" + ".html")
# %%
