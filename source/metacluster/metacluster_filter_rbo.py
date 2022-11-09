# %%
import os
import sys
sys.path.append("./")
sys.setrecursionlimit(10000)
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from settings import params, paths

figPath = paths.outputDir + "rnaseq/metacluster_10pct_rbo_noPol/"
usePolII = False
try:
    os.mkdir(figPath)
except FileExistsError:
    pass
mat = []
names = []
# %%
from scipy.stats import rankdata
filesEncode = os.listdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "Encode_" + f[4:-4]
    resFull = pd.read_csv(paths.outputDir + "rnaseq/encode_rnaseq/DE/" + f, index_col=0)
    resFull = resFull[resFull["Upreg"] > 0.5]
    resFull.sort_values(["pval", "stat"], ascending=[True, False], inplace=True)
    if np.sum(resFull["Upreg"].values) < 100:
        continue
    mat.append(resFull.index.astype("str").values)
    names.append(name)
    

# %%
if usePolII:
    filesPol2 = os.listdir(paths.outputDir + "enrichedPerAnnot/")
    filesPol2 = [f for f in filesPol2 if f.endswith("_qvals.bed")]
    for f in filesPol2:
        name = "Pol2_" + f[:-10]
        try:
            resFull = pd.read_csv(paths.outputDir + "enrichedPerAnnot/" + f, sep="\t", index_col=0)
            resFull = resFull[resFull["qval"] < 0.05]
            if len(resFull) < 100:
                continue
            resFull.sort_values(["pval", "fc"], ascending=[True, False], inplace=True)
            mat.append(resFull.index.astype("str").values)
            names.append(name)
        except pd.errors.EmptyDataError:
            continue 
# matPolII = pd.DataFrame(matEncode, index=[f"Pol2_f[4:-4]" for f in filesEncode])
# %%
# Files GTex
filesEncode = os.listdir(paths.outputDir + "rnaseq/gtex_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "GTEx_" + f[4:-4]
    resFull = pd.read_csv(paths.outputDir + "rnaseq/gtex_rnaseq/DE/" + f, index_col=0)
    resFull = resFull[resFull["Upreg"] > 0.05]
    resFull.sort_values(["pval", "stat"], ascending=[True, False], inplace=True)
    if np.sum(resFull["Upreg"].values) < 100:
        continue
    mat.append(resFull.index.astype("str").values)
    names.append(name)
    

# %%
# Files TCGA
filesEncode = os.listdir(paths.outputDir + "rnaseq/TCGA/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "TCGA_" + f[4:-4]
    resFull = pd.read_csv(paths.outputDir + "rnaseq/TCGA/DE/" + f, index_col=0)
    resFull = resFull[resFull["Upreg"] > 0.5]
    resFull.sort_values(["pval", "stat"], ascending=[True, False], inplace=True)
    if np.sum(resFull["Upreg"].values) < 100:
        continue
    mat.append(resFull.index.astype("str").values)
    names.append(name)

# %%
import rbo
from sklearn.metrics import ndcg_score
# Build matrix
dst = pd.DataFrame(np.zeros((len(mat), len(mat))), columns=names, index=names)
for i in range(len(mat)):
    for j in range(len(mat)):
        if j >= i:
            dst.iloc[i, j] = 1-rbo.RankingSimilarity(mat[i], mat[j]).rbo_ext(p=0.998)
dst.loc[:] = np.clip(np.maximum(dst, dst.transpose()),0,1)
# %%
# UMAP
import umap
from lib.utils import plot_utils
embedding = umap.UMAP(n_neighbors=10, min_dist=0.0, metric="precomputed", verbose=True, n_epochs=5000, random_state=42).fit_transform(dst.values)
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
tissue = pd.Series(dst.index).str.split("_", expand=True)
df[["Orig", "Annot", "State"]] = tissue
df["Annot"] = df["Annot"].str.replace("-", "/")
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
linkage = fastcluster.linkage(dst, "average")
row_order = hierarchy.leaves_list(linkage)
# dst = -sd.squareform(sd.pdist(mat, metric))
# dst = pd.DataFrame(dst, columns=mat.index, index=mat.index)
dst2 = np.sqrt(1-dst.copy())
dst2 = dst2.iloc[row_order].iloc[:, row_order]
# Plot
fig = px.imshow(dst2, width=2580, height=1440)
fig.update_layout(yaxis_nticks=len(dst2),
                  xaxis_nticks=len(dst2))
colsInt = np.array(colors*255).astype(int)[row_order]
ticktext = [color(rgb2hex(tuple(c)), t) for c, t in zip(colsInt, dst2.columns)]
fig.update_layout(yaxis=dict(tickmode='array', ticktext=ticktext, tickvals=np.arange(len(dst2))))
fig.update_layout(xaxis=dict(tickmode='array', ticktext=ticktext, tickvals=np.arange(len(dst2))))
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
# Pol 2 Matching vs non-matching
if usePolII:
    import matplotlib.pyplot as plt
    import seaborn as sns
    vals = []
    tissues = pd.Series(dst.index).str.split("_", expand=True)
    for i in range(len(dst)):
        for j in range(1+i, len(dst)):
            if tissues[0][i] == "Pol2":
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
    plt.savefig(figPath + "yule_matching_pol2.pdf")
    plt.show()
    fig = px.strip(data_frame=yuleDf, x="Annotation", y="Yule coefficient",hover_data=["M1", "M2"])
    fig.show()
# %%
dst.to_csv(figPath + "heatmapMetacluster.csv")
# %%
# UMAP 3D
import umap
from lib.utils import plot_utils
embedding = umap.UMAP(10, n_components=3, min_dist=0.0, metric="precomputed", n_epochs=5000,random_state=42).fit_transform(dst)
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y","z"])
tissue = pd.Series(dst.index).str.split("_", expand=True)
df[["Orig", "Annot", "State"]] = tissue
df["Annot"] = df["Annot"].str.replace("-", "/")
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
fig = px.bar(x=np.array(names)[row_order], y=np.array([len(m) for m in mat])[row_order], width=2000)
fig.update_layout(xaxis=dict( tickvals=np.arange(len(dst))))
fig.show()
fig.write_image(figPath + "n_markers.pdf")
fig.write_html(figPath + "n_markers.pdf" + ".html")
# %%
