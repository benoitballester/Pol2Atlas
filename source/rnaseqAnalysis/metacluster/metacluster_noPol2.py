# %%
import os
import sys
sys.path.append("./")
sys.setrecursionlimit(10000)
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from settings import paths

figPath = paths.outputDir + "rnaseq/metacluster_10pct_topK_noPol/"
try:
    os.mkdir(figPath)
except FileExistsError:
    pass
indices = dict()
topK = 1000
# %%
filesEncode = os.listdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "Encode_" + f[4:-4]
    resFull = pd.read_csv(paths.outputDir + "rnaseq/encode_rnaseq/DE/" + f, index_col=0)
    resFull = resFull[resFull["Upreg"] == 1]
    resFull.sort_values(["pval", "stat"], ascending=[True, False], inplace=True)
    res = pd.read_csv(paths.outputDir + "rnaseq/encode_rnaseq/DE/" + f, index_col=0)["Upreg"] 
    vals = resFull.index[:topK].values
    if len(vals) < 100:
        continue
    indices[name] = vals

# %%

# Files GTex
filesEncode = os.listdir(paths.outputDir + "rnaseq/gtex_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "GTex_" + f[4:-4]
    resFull = pd.read_csv(paths.outputDir + "rnaseq/gtex_rnaseq/DE/" + f, index_col=0)
    resFull = resFull[resFull["Upreg"] == 1]
    resFull.sort_values(["pval", "stat"], ascending=[True, False], inplace=True)
    res = pd.read_csv(paths.outputDir + "rnaseq/gtex_rnaseq/DE/" + f, index_col=0)["Upreg"] 
    vals = resFull.index[:topK].values
    if len(vals) == 0:
        continue
    indices[name] = vals

# %%
# Files TCGA
filesEncode = os.listdir(paths.outputDir + "rnaseq/TCGA/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "TCGA_" + f[4:-4]
    if name == "TCGA_nan":
        continue
    resFull = pd.read_csv(paths.outputDir + "rnaseq/TCGA/DE/" + f, index_col=0)
    resFull = resFull[resFull["Upreg"] == 1]
    resFull.sort_values(["pval", "stat"], ascending=[True, False], inplace=True)
    res = pd.read_csv(paths.outputDir + "rnaseq/TCGA/DE/" + f, index_col=0)["Upreg"] 
    vals = resFull.index[:topK].values
    if res.sum() == 0:
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
mat = mat.loc[mat.sum(axis=1)>100]
tf = mat.values / mat.values.sum(axis=1).reshape(-1, 1)
tf = tf[:, (mat.values.mean(axis=0) <= 0.1) & (mat.values.sum(axis=0) >= 2)]
mat = mat.loc[:, (mat.mean(axis=0) <= 0.1) & (mat.sum(axis=0) >= 2)]

from sklearn.feature_extraction.text import TfidfTransformer
mat
# %%
# UMAP
import umap
from lib.utils import plot_utils
embedding = umap.UMAP(n_neighbors=10, min_dist=0.0, metric="yule", verbose=True, n_epochs=5000, random_state=42).fit_transform(mat)
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
tissue = pd.Series(mat.index).str.split("_", expand=True)
df[["Orig", "Annot", "State"]] = tissue
annot, palette, colors = plot_utils.applyPalette(df["Annot"],
                                                np.unique(df["Annot"]),
                                                 paths.polIIannotationPalette, ret_labels=True)
palettePlotly = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in palette]
colormap = dict(zip(annot, palettePlotly))     
# %%
import plotly.graph_objects as go
all_figs = []
for c in df["Annot"]:
    tagged = embedding[df["Annot"]==c]
    j = 0
    k = 0
    for i in range(0, int((len(tagged)**2-len(tagged))/2)):
        j += 1
        print(i, j, k)
        fig1 = px.line(x=tagged[[j,k], 0], y=tagged[[j,k], 1])
        fig1.update_traces(line=dict(color="rgba" + colormap[c][3:-1] + ",0.5)", width=2))
        all_figs.append(fig1)
        if j == len(tagged)-1:
            k += 1
            j = k
fig = px.scatter(df, x="x", y="y", color="Annot", color_discrete_map=colormap, symbol="Orig",
                hover_data=['Orig', "State"], width=1200, height=800)
fig.update_traces(marker=dict(size=100/np.sqrt(len(df))))
fig.show()
fig.write_image(figPath + "metacluster_umap.pdf")
fig.write_html(figPath + "metacluster_umap.pdf" + ".html")
import operator
import functools
fig3 = go.Figure(data=functools.reduce(operator.add, [_.data for _ in all_figs]) + fig.data)
fig3.update_layout(
    autosize=False,
    width=1200,
    height=800)
fig3.show()

fig3.write_image(figPath + "metacluster_umap_lines.pdf")
fig3.write_html(figPath + "metacluster_umap_lines.pdf" + ".html")
# %%
# Heatmap
metric='yule'
from lib.utils import matrix_utils
import scipy.spatial.distance as sd
import fastcluster
from scipy.cluster import hierarchy

# Need HTML + hexadecimal ticks for plotly
def rgb2hex(rgb):
    print(rgb)
    return '#%02x%02x%02x' % rgb
    
def color(color, text):
    # color: hexadecimal
    s = "<span style='color:" + str(color) + "'>" + str(text) + "</span>"
    return s

# Complete linkage hc
linkage = fastcluster.linkage(mat, "average", metric)
row_order = hierarchy.leaves_list(linkage)
dst = -sd.squareform(sd.pdist(mat, metric))
dst = pd.DataFrame(dst, columns=mat.index, index=mat.index)
dst = dst.iloc[row_order].iloc[:, row_order]
# Plot
fig = px.imshow(dst, width=2580, height=1440)
fig.update_layout(yaxis_nticks=len(dst),
                  xaxis_nticks=len(dst))
colsInt = np.array(colors*255).astype(int)[row_order]
ticktext = [color(rgb2hex(tuple(c)), t) for c, t in zip(colsInt, dst.columns)]
fig.update_layout(yaxis=dict(tickmode='array', ticktext=ticktext, tickvals=np.arange(len(dst))))
fig.update_layout(xaxis=dict(tickmode='array', ticktext=ticktext, tickvals=np.arange(len(dst))))
fig.show()
fig.write_image(figPath + "metacluster.pdf")
fig.write_html(figPath + "metacluster.pdf" + ".html")
# %%
# Matching vs non-matching
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
vals = []
tissues = pd.Series(dst.index).str.split("_", expand=True)
for i in range(len(dst)):
    for j in range(1+i, len(dst)):
        if tissues[1][i] == tissues[1][j]:
            print(dst.index[i], dst.columns[j])
            vals.append([dst.iloc[i,j], "Matching", dst.index[i], dst.index[j]])
        else:
            vals.append([dst.iloc[i,j],"Not matching", dst.index[i], dst.index[j]])
yuleDf = pd.DataFrame(vals, columns=["Yule coefficient", "Annotation", "M1", "M2"])
plt.figure(dpi=500)
sns.boxplot(data=yuleDf, x="Annotation", y="Yule coefficient", showfliers=False)
sns.stripplot(data=yuleDf, x="Annotation", y="Yule coefficient", s=1.0, dodge=True, jitter=0.4, linewidths=1.0)
pval = mannwhitneyu(yuleDf['Yule coefficient'][yuleDf['Annotation'] == 'Not matching'], yuleDf['Yule coefficient'][yuleDf['Annotation'] == 'Matching'])[1]
plt.title(f"pval={pval}")
plt.savefig(figPath + "yule_matching.pdf")
plt.show()
# %%
fig = px.strip(data_frame=yuleDf, x="Annotation", y="Yule coefficient",hover_data=["M1", "M2"])
fig.show()
# %%
dst.to_csv(figPath + "heatmapMetacluster.csv")
# %%
# UMAP 3D
import umap
from lib.utils import plot_utils
embedding = umap.UMAP(10, n_components=3, min_dist=0.0, metric="yule", n_epochs=5000,random_state=42).fit_transform(mat)
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y","z"])
tissue = pd.Series(mat.index).str.split("_", expand=True)
df[["Orig", "Annot", "State"]] = tissue
annot, palette, colors = plot_utils.applyPalette(df["Annot"],
                                                np.unique(df["Annot"]),
                                                 paths.polIIannotationPalette, ret_labels=True)
palettePlotly = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in palette]
colormap = dict(zip(annot, palettePlotly))     
# %%
import plotly.graph_objects as go
all_figs = []
for c in df["Annot"]:
    tagged = embedding[df["Annot"]==c]
    j = 0
    k = 0
    for i in range(0, int((len(tagged)**2-len(tagged))/2)):
        j += 1
        print(i, j, k)
        fig1 = px.line_3d(x=tagged[[j,k], 0], y=tagged[[j,k], 1], z=tagged[[j,k], 2])
        fig1.update_traces(line=dict(color="rgba" + colormap[c][3:-1] + ",0.5)", width=2))
        all_figs.append(fig1)
        if j == len(tagged)-1:
            k += 1
            j = k
fig = px.scatter_3d(df, x="x", y="y", z="z", color="Annot", color_discrete_map=colormap, symbol="Orig",
                hover_data=['Orig', "State"], width=1200, height=800)
fig.update_traces(marker=dict(size=50/np.sqrt(len(df))))
fig.show()
fig.write_html(figPath + "metacluster_umap3d.pdf" + ".html")
import operator
import functools
fig3 = go.Figure(data=functools.reduce(operator.add, [_.data for _ in all_figs]) + fig.data)
fig3.update_layout(
    autosize=False,
    width=1200,
    height=800)
fig3.show()
fig3.write_html(figPath + "metacluster_umap3d_lines.pdf" + ".html")
# %%
fig = px.bar(x=mat.index[row_order], y=mat.sum(axis=1)[row_order], width=2000)
fig.update_layout(xaxis=dict( tickvals=np.arange(len(dst))))
fig.show()
fig.write_image(figPath + "n_markers.pdf")
fig.write_html(figPath + "n_markers.pdf" + ".html")
# %%
