# %%
import os
import sys
sys.path.append("./")
sys.setrecursionlimit(10000)
import numpy as np
import pandas as pd
from lib import rnaseqFuncs
from statsmodels.stats.multitest import fdrcorrection
from settings import paths

indices = dict()
# %%
filesEncode = os.listdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "Encode_" + f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/encode_rnaseq/DE/" + f, index_col=0)["Upreg"] 
    vals = res.index[res==1].values
    if len(vals) == 0:
        continue
    indices[name] = vals

# %%
filesPol2 = os.listdir(paths.outputDir + "enrichedPerAnnot/")
filesPol2 = [f for f in filesPol2 if not f.endswith("_qvals.bed")]
for f in filesPol2:
    name = "Pol2_" + f[:-4]
    try:
        bed = pd.read_csv(paths.outputDir + "enrichedPerAnnot/" + f, header=None, sep="\t")
        indices[name] = bed[3].values
    except pd.errors.EmptyDataError:
        continue
# matPolII = pd.DataFrame(matEncode, index=[f"Pol2_f[4:-4]" for f in filesEncode])
# %%
# Files GTex
filesEncode = os.listdir(paths.outputDir + "rnaseq/gtex_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "GTex_" + f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/gtex_rnaseq/DE/" + f, index_col=0)["Upreg"] 
    indices[name] = res.index[res==1].values
    vals = res.index[res==1].values
    if len(vals) == 0:
        continue
    indices[name] = vals
# %%
# Build matrix
from scipy.sparse import csr_matrix
rows = []
cols = []
data = []
for i, k in enumerate(indices.keys()):
    idx = list(indices[k])
    cols += idx
    rows += [i]*len(idx)
    data += [True]*len(idx)
mat = csr_matrix((data, (rows, cols)), shape=(len(indices), np.max(cols)+1), dtype=bool)
mat = pd.DataFrame(mat.todense(), index=indices.keys())

# %%
# UMAP
import umap
from sklearn.manifold import MDS
embedding = umap.UMAP(min_dist=0.0, metric="yule", random_state=42).fit_transform(mat)
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
df[["Orig", "Annot"]] = np.array([m.split("_") for m in mat.index])
fig = px.scatter(df, x="x", y="y", color="Annot",
                hover_data=['Orig'], width=1200, height=800)
fig.update_traces(marker=dict(size=100/np.sqrt(len(df))))
fig.show()
# %%
# Heatmap
metric='yule'
from lib.utils import matrix_utils
import scipy.spatial.distance as sd
import fastcluster
from scipy.cluster import hierarchy
linkage = fastcluster.linkage(mat, "average", metric)
row_order = hierarchy.leaves_list(linkage)
dst = -sd.squareform(sd.pdist(mat, metric))
dst = pd.DataFrame(dst, columns=mat.index, index=mat.index)
dst = dst.iloc[row_order].iloc[:, row_order]
fig = px.imshow(dst, width=1200, height=800)
fig.update_layout(yaxis_nticks=len(dst),
                  xaxis_nticks=len(dst))
fig.show()
# %%
