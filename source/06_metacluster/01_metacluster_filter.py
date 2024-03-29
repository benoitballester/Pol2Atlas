# %%
import os
import sys
sys.path.append("./")
sys.setrecursionlimit(10000)
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from settings import params, paths

figPath = paths.outputDir + "rnaseq/metacluster_10pct/"
try:
    os.mkdir(figPath)
except FileExistsError:
    pass
indices = dict()
# %%
filesEncode = os.listdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "Encode_" + f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/encode_rnaseq/DE/" + f, index_col=0)["Upreg"] 
    vals = res.index[res==1].values
    if len(vals) < 100:
        continue
    indices[name] = vals

# %%
filesPol2 = os.listdir(paths.outputDir + "enrichedPerAnnot/")
filesPol2 = [f for f in filesPol2 if not f.endswith("_qvals.bed")]
for f in filesPol2:
    name = "Pol2_" + f[:-4]
    try:
        bed = pd.read_csv(paths.outputDir + "enrichedPerAnnot/" + f, header=None, sep="\t")
        if len(bed) < 100:
            continue
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
    vals = res.index[res==1].values
    if len(vals) == 0:
        continue
    indices[name] = vals


# Files TCGA
filesEncode = os.listdir(paths.outputDir + "rnaseq/TCGA2/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "TCGA_" + f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/TCGA2/DE/" + f, index_col=0)["Upreg"] 
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
subsetGTex = mat.loc[mat.index.str.startswith("GTex")]
mat = mat.loc[mat.sum(axis=1) > 100]
mat = mat.loc[:, (mat.mean(axis=0) <= 0.1) & (mat.sum(axis=0) >= 2)]
from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer(norm=None, smooth_idf=False).fit_transform(mat.values).toarray()
# %%
# UMAP
import umap
from lib.utils import plot_utils
embedding = umap.UMAP(n_neighbors=10, min_dist=0.0, metric="yule", verbose=True, n_epochs=5000, random_state=42).fit_transform(mat)
import plotly.express as px
df = pd.DataFrame(embedding, columns=["x","y"])
tissue = pd.Series(mat.index).str.split("_", expand=True)
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
idf = np.log(mat.sum(axis=0).values)
dst = -sd.squareform(sd.pdist(mat, metric))
linkage = fastcluster.linkage(-sd.squareform(dst), "average", metric)
row_order = hierarchy.leaves_list(linkage)
dst = pd.DataFrame(dst, columns=mat.index, index=mat.index)
dst = dst.iloc[row_order].iloc[:, row_order]
# Plot
""" fig = px.imshow(dst, width=2580, height=1440)
fig.update_layout(yaxis_nticks=len(dst),
                  xaxis_nticks=len(dst))
colsInt = np.array(colors*255).astype(int)[row_order]
ticktext = [color(rgb2hex(tuple(c)), t) for c, t in zip(colsInt, dst.columns)]
fig.update_layout(yaxis=dict(tickmode='array', ticktext=ticktext, tickvals=np.arange(len(dst))))
fig.update_layout(xaxis=dict(tickmode='array', ticktext=ticktext, tickvals=np.arange(len(dst))))
fig.show()
fig.write_image(figPath + "metacluster.pdf")
fig.write_html(figPath + "metacluster.pdf" + ".html") """
# %%
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(dpi=500)
ax=sns.clustermap(dst, row_cluster=False, col_cluster=False,
                  row_colors=colors[row_order],col_colors=colors[row_order], 
                  linewidth=0, xticklabels=True, 
                  yticklabels=True, cmap="rocket")
ax.ax_heatmap.axes.tick_params(right=False, bottom=False)
plt.tight_layout()
plt.gca().set_aspect(1.0)
for ticklabel, tickcolor in zip(ax.ax_heatmap.axes.get_xticklabels(), colors[row_order]):
    ticklabel.set_color(tickcolor)
for ticklabel, tickcolor in zip(ax.ax_heatmap.axes.get_yticklabels(), colors[row_order]):
    ticklabel.set_color(tickcolor)
ticks = [ticklabel.get_text().replace("GTex", "GTEx").replace("Encode", "ENCODE") for ticklabel in ax.ax_heatmap.axes.get_xticklabels()]
ax.ax_heatmap.axes.set_xticklabels(ticks,fontsize=5)
ax.ax_heatmap.axes.set_yticklabels(ticks,fontsize=5)
ax.ax_heatmap.axes.tick_params(axis='both', which='major', pad=-2)
box_heatmap = ax.ax_heatmap.get_position()
ax_row_colors = ax.ax_row_colors
box = ax_row_colors.get_position()
ax_row_colors.set_position([box_heatmap.x0-box.width, box.y0, box.width, box_heatmap.height])
ax_col_colors = ax.ax_col_colors
box = ax_col_colors.get_position()
ax_col_colors.set_position([box_heatmap.x0, box_heatmap.y1, box.width, box.height])
plt.savefig(figPath + "metacluster_sns.pdf")
plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(dpi=500)
ax=sns.clustermap(dst, row_cluster=False, col_cluster=False,
                  row_colors=colors[row_order],col_colors=colors[row_order], 
                  linewidth=0, xticklabels=True, 
                  yticklabels=True, cmap="viridis")
ax.ax_heatmap.axes.tick_params(right=False, bottom=False)
plt.tight_layout()
plt.gca().set_aspect(1.0)
for ticklabel, tickcolor in zip(ax.ax_heatmap.axes.get_xticklabels(), colors[row_order]):
    ticklabel.set_color(tickcolor)
for ticklabel, tickcolor in zip(ax.ax_heatmap.axes.get_yticklabels(), colors[row_order]):
    ticklabel.set_color(tickcolor)
ticks = [ticklabel.get_text().replace("GTex", "GTEx").replace("Encode", "ENCODE") for ticklabel in ax.ax_heatmap.axes.get_xticklabels()]
ax.ax_heatmap.axes.set_xticklabels(ticks,fontsize=5)
ax.ax_heatmap.axes.set_yticklabels(ticks,fontsize=5)
ax.ax_heatmap.axes.tick_params(axis='both', which='major', pad=-2)
box_heatmap = ax.ax_heatmap.get_position()
ax_row_colors = ax.ax_row_colors
box = ax_row_colors.get_position()
ax_row_colors.set_position([box_heatmap.x0-box.width, box.y0, box.width, box_heatmap.height])
ax_col_colors = ax.ax_col_colors
box = ax_col_colors.get_position()
ax_col_colors.set_position([box_heatmap.x0, box_heatmap.y1, box.width, box.height])
plt.savefig(figPath + "metacluster_sns_ben_gradient_bizarre.pdf")
plt.show()
# %%
# Matching vs non-matching
from scipy.stats import mannwhitneyu
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
fig = px.bar(x=mat.index[row_order], y=mat.sum(axis=1)[row_order], width=2000)
fig.update_layout(xaxis=dict( tickvals=np.arange(len(dst))))
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
fig.show()
fig.write_image(figPath + "n_markers.pdf")
fig.write_html(figPath + "n_markers.pdf" + ".html")
# %%
fig = px.bar(x=mat.index[row_order], y=mat.sum(axis=1)[row_order], width=2000)
fig.update_layout(xaxis=dict( tickvals=np.arange(len(dst))))
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
fig.show()
fig.write_image(figPath + "n_markers_nobg.pdf")
fig.write_html(figPath + "n_markers_nobg.pdf" + ".html")
# %%
fig = px.bar(x=mat.index[row_order], y=mat.sum(axis=1)[row_order], width=2000, log_y=True)
fig.update_layout(xaxis=dict( tickvals=np.arange(len(dst))))
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
fig.show()
fig.write_image(figPath + "n_markers_nobg_logy.pdf")
fig.write_html(figPath + "n_markers_nobg_logy.pdf" + ".html")
# %%
try:
    os.mkdir(figPath + "markerCount/")
except FileExistsError:
    pass
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None, usecols=[0,1,2,3,4])
for t in np.unique(tissue[1].values):
    copy = consensuses.copy()
    copy[4] = 0
    nMarkers = mat[tissue[1].values == t].sum(axis=0)
    copy.loc[nMarkers.index, 4] = nMarkers.values
    copy.to_csv(figPath + f"markerCount/allwithCounts_{t.replace(' ','_')}.bed", header=None, index=None, sep="\t")
    copy[4] = 0
    nMarkers = mat[tissue[1].values == t].mean(axis=0)
    copy[4] = 0
    copy.loc[nMarkers.index, 4] = nMarkers.values
    copy = copy[copy[4] > 0.99]
    copy[4] = copy[4].astype(int)
    copy.to_csv(figPath + f"markerCount/allDatasets_{t.replace(' ','_')}.bed", header=None, index=None, sep="\t")
    copy = consensuses.copy()
    copy[4] = 0
    copy.loc[nMarkers.index, 4] = nMarkers.values
    copy = copy[copy[4] >= 0.5]
    copy[4] = copy[4]
    copy.to_csv(figPath + f"markerCount/halfDatasets_{t.replace(' ','_')}.bed", header=None, index=None, sep="\t")
    copy = consensuses.copy()
    copy[4] = 0
    nMarkers = mat[tissue[1].values == t].sum(axis=0).astype(int)
    copy.loc[nMarkers.index, 4] = nMarkers.values
    copy = copy[copy[4] >= 2]
    if len(copy) > 0:
        copy.to_csv(figPath + f"markerCount/min2_{t.replace(' ','_')}.bed", header=None, index=None, sep="\t")
# %%
with open(paths.tempDir + "end0601.txt", "w") as f:
    f.write("1")