# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pyranges as pr
import seaborn as sns
from lib.utils import overlap_utils
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom, mannwhitneyu
from settings import params, paths

try:
    os.mkdir(paths.outputDir + "eqtlAnalysis/")
except FileExistsError:
    pass

pctSpecEqtl = 0.1
pctSpecMarker = 0.1
eqtlPath = paths.gtexData + "eQTL/GTEx_Analysis_v8_eQTL/"
eqtlFiles = os.listdir(eqtlPath)
eqtlFiles = [f for f in eqtlFiles if f.endswith("signif_variant_gene_pairs.txt.gz")]
# %%
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None, usecols=[0,1,2,3])
consensuses.columns = ["Chromosome", "Start", "End", "Name"]
# Read per tissue markers
indices = dict()
# Files GTex
filesEncode = os.listdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/encode_rnaseq/DE/" + f, index_col=0)["Upreg"] 
    vals = res.index[res==1].values
    if len(vals) < 100:
        continue
    indices[name] = vals
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
mat = csr_matrix((data, (rows, cols)), shape=(len(indices), len(consensuses)), dtype=bool)
mat = pd.DataFrame(mat.todense(), index=indices.keys())
# Throw away non-markers and low specificity markers
minSpec = max(int(mat.shape[0]*pctSpecEqtl+0.5),1)
kept = (mat.sum(axis=0) <= minSpec).values & (mat.sum(axis=0) != 0).values
consensuses = consensuses.iloc[kept]
mat = mat.iloc[:, kept]
# Remove columns with low marker count (50)
keptAnnots = (mat.sum(axis=1) > 100).values
mat = mat.iloc[keptAnnots]

# %%
'''
tissue = eqtlFiles[7].split(".")[0].split("_")[0]
df = pd.read_csv(eqtlPath + eqtlFiles[7], sep="\t", usecols=[0,1])
dfBed = df["variant_id"].str.split("_", expand=True).iloc[:, :2]
dfBed.columns = ["Chromosome", "Start"]
dfBed["Start"] = dfBed["Start"].astype("int")
dfBed["End"] = dfBed["Start"] + 1
dfBed["Name"] = df["gene_id"]
dfBed = pr.PyRanges(dfBed)
tissueMarkers = pr.PyRanges(consensuses[mat.loc["Heart"]])
from scipy.stats import hypergeom
k = len(tissueMarkers.overlap(dfBed))
n = len(pr.PyRanges(consensuses).overlap(dfBed))
M = len(consensuses)
N = len(tissueMarkers)
p = hypergeom(M, n, N).sf(k-1)
fc = (k/N)/(n/M)
print(tissue, fc, p)
'''
# %%
# Read eqtl files and store eQTL-gene association vs observed in tissue in a binary matrix
tissueEqtls = dict()
allEqtls = np.array([])
tissues = set()
for f in eqtlFiles:
    tissue = f.split(".")[0]
    df = pd.read_csv(eqtlPath + f, sep="\t", usecols=[0,1])
    eqtls = (df["variant_id"]).values
    tissueEqtls[tissue] = eqtls
    allEqtls = np.unique(np.concatenate([eqtls, allEqtls]))
    tissues.add(tissue)
eqtlDf = pd.DataFrame(np.zeros((len(allEqtls), len(tissues)), "bool"), index=allEqtls, columns=tissues)
for i in tissues:
    eqtlDf.loc[tissueEqtls[i], i] = True
# %%
colors = pd.read_csv(paths.gtexData + "colors.txt", 
                        sep="\t", index_col="tissue_id")
tissues = set(colors.loc[tissues, "tissue_site_detail"])
eqtlDf.columns = colors.loc[eqtlDf.columns, "tissue_site_detail"]
# %%
# Trim low-specificity eQTL-gene associations, generate bed file
minSpec = max(int(eqtlDf.shape[1]*pctSpecMarker+0.5),1)
highSpec = (eqtlDf.sum(axis=1) <= minSpec).values
eqtlPos = pd.Series(allEqtls).str.split("_", expand=True)[[0,1]]
eqtlPos.columns = [["Chromosome", "Start"]]
eqtlPos["Start"] = eqtlPos["Start"].astype("int")
eqtlPos["End"] = eqtlPos["Start"] + 1

# %%
# Build matrix
def buildMarkerMatrix(consensuses, indices):
    rows = []
    cols = []
    data = []
    for i, k in enumerate(indices.keys()):
        idx = list(indices[k])
        cols += idx
        rows += [i]*len(idx)
        data += [True]*len(idx)
    mat = csr_matrix((data, (rows, cols)), shape=(len(indices), len(consensuses)), dtype=bool)
    mat = pd.DataFrame(mat.todense(), index=indices.keys())
    # Throw away non-markers and low specificity markers
    minSpec = max(int(mat.shape[0]*pctSpecEqtl+0.5),1)
    kept = (mat.sum(axis=0) <= minSpec).values & (mat.sum(axis=0) != 0).values
    consensuses = consensuses.iloc[kept]
    mat = mat.iloc[:, kept]
    # Remove columns with low marker count (50)
    keptAnnots = (mat.sum(axis=1) > 100).values
    mat = mat.iloc[keptAnnots]
    return consensuses,mat
def rgb2hex(rgb):
    print(rgb)
    return '#%02x%02x%02x' % rgb
def color(color, text):
    # color: hexadecimal
    s = "<span style='color:" + str(color) + "'>" + str(text) + "</span>"
    return s
# Compute enrichments of per tissue eqtl in per tissue Pol II markers
def eqtlEnrich(consensuses, mat, tissues, eqtlDf, highSpec, eqtlPos, name, colors):
    mat2 = mat.copy()
    mat2 = mat2.loc[eqtlDf.columns]
    print(mat2)
    fcDf = pd.DataFrame(index=tissues, columns=mat2.index)
    pvalDf = pd.DataFrame(index=tissues, columns=mat2.index)
    for i in tissues:
            eqtlPR = overlap_utils.dfToPrWorkaround(eqtlPos[highSpec & eqtlDf[i].values], False)
            for j in mat2.index:
                tissueMarkers = pr.PyRanges(consensuses[mat2.loc[j]])
                k = len(tissueMarkers.overlap(eqtlPR))
                n = len(pr.PyRanges(consensuses).overlap(eqtlPR))
                M = len(consensuses)
                N = len(tissueMarkers)
                p = hypergeom(M, n, N).sf(k-1)
                fc = ((k+1e-9)/(1e-9+N))/((1e-9+n)/(1e-9+M))
                fcDf.loc[i, j] = fc
                pvalDf.loc[i, j] = p
    matchingAnnot = list(tissues)
    fcDf = fcDf.astype("float")
    fcDf.fillna(1.0, inplace=True)
    pvalDf = pvalDf.astype("float")
    pvalDf.fillna(1.0, inplace=True)
    orderCol = np.argsort(mat2.index)
    orderRow = np.argsort(matchingAnnot)
    fig = px.imshow(-np.log10(pvalDf).iloc[orderRow].iloc[:, orderCol], zmax=np.percentile(-np.log10(pvalDf), 99), 
                width=1500, height=1500)
    fig.update_layout(yaxis_nticks=len(tissues),
                  xaxis_nticks=len(mat2.index))
    colors.index = colors["tissue_site_detail"]
    ticktext = [color(c, t) for c, t in zip(colors.loc[pvalDf.index[orderRow], "color_hex"], pvalDf.index[orderRow])]
    ticktextRow = [color(c, t[:10]+"...") for c, t in zip(colors.loc[pvalDf.index[orderRow], "color_hex"], pvalDf.index[orderRow])]
    fig.update_layout(yaxis=dict(tickmode='array', tickfont=dict(size=24), ticktext=ticktext, tickvals=np.arange(len(mat2.index))))
    fig.update_layout(xaxis=dict(tickmode='array', tickfont=dict(size=24), ticktext=ticktextRow, tickvals=np.arange(len(mat2.index))))
    fig.show()
    fig.write_image(paths.outputDir + f"eqtlAnalysis/heatmap{name}.pdf")
    fig.write_html(paths.outputDir + f"eqtlAnalysis/heatmap{name}.pdf" + ".html")
    # Matching vs non-matching
    vals = []
    for i in tissues:
        matchingAnnot = i
        for j in mat2.index:
            if matchingAnnot == j.split("_")[0]:
                print(i, matchingAnnot, j)
                vals.append([pvalDf.loc[i,j], "Matching", i, j])
            else:
                vals.append([pvalDf.loc[i,j],"Not matching", i, j])
    yuleDf = pd.DataFrame(vals, columns=["Enrichment p-value", "Annotation", "M1", "M2"])
    plt.figure(dpi=500)
   
    sns.boxplot(data=yuleDf, x="Annotation", y="Enrichment p-value", showfliers=False)
    sns.stripplot(data=yuleDf, x="Annotation", y="Enrichment p-value", hue="Annotation",
                jitter=0.4, s=2.0, linewidth=0.1, edgecolor="black")
    pval = mannwhitneyu(yuleDf['Enrichment p-value'][yuleDf['Annotation'] == 'Not matching'], yuleDf['Enrichment p-value'][yuleDf['Annotation'] == 'Matching'])[1]
    plt.legend([],[], frameon=False)
    plt.title(f"Mann-Whitney p-value={pval:.2E}")
    plt.savefig(paths.outputDir + f"eqtlAnalysis/boxplot{name}.pdf")
    plt.show()

# GTEX
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None, usecols=[0,1,2,3])
consensuses.columns = ["Chromosome", "Start", "End", "Name"]
# Read per tissue markers
indices = dict()
filesEncode = os.listdir(paths.outputDir + "rnaseq/gtex_rnaseq/DE54/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/gtex_rnaseq/DE54/" + f, index_col=0)["Upreg"] 
    vals = res.index[res==1].values
    if len(vals) == 0:
        continue
    indices[name] = vals
consensuses, mat = buildMarkerMatrix(consensuses, indices)
eqtlEnrich(consensuses, mat, tissues, eqtlDf, highSpec, eqtlPos, "GTEx", colors)
# %%
'''
# Pol II
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None, usecols=[0,1,2,3])
consensuses.columns = ["Chromosome", "Start", "End", "Name"]
indices = dict()
# Read per tissue markers
filesPol2 = os.listdir(paths.outputDir + "enrichedPerAnnot/")
filesPol2 = [f for f in filesPol2 if not f.endswith("_qvals.bed")]
for f in filesPol2:
    name = f[:-4]
    try:
        bed = pd.read_csv(paths.outputDir + "enrichedPerAnnot/" + f, header=None, sep="\t")
        if len(bed) < 100:
            continue
        indices[name] = bed[3].values
    except pd.errors.EmptyDataError:
        continue

consensuses, mat = buildMarkerMatrix(consensuses, indices)
eqtlEnrich(consensuses, mat, tissues, eqtlDf, highSpec, eqtlPos, "PolII")

# TCGA
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None, usecols=[0,1,2,3])
consensuses.columns = ["Chromosome", "Start", "End", "Name"]
indices = dict()
# Read per tissue markers
filesEncode = os.listdir(paths.outputDir + "rnaseq/TCGA/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/TCGA/DE/" + f, index_col=0)["Upreg"] 
    vals = res.index[res==1].values
    if len(vals) == 0:
        continue
    indices[name] = vals


consensuses, mat = buildMarkerMatrix(consensuses, indices)
eqtlEnrich(consensuses, mat, tissues, eqtlDf, highSpec, eqtlPos, "TCGA")

# ENCODE
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None, usecols=[0,1,2,3])
consensuses.columns = ["Chromosome", "Start", "End", "Name"]
# Read per tissue markers
indices = dict()
filesEncode = os.listdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/encode_rnaseq/DE/" + f, index_col=0)["Upreg"] 
    vals = res.index[res==1].values
    if len(vals) < 100:
        continue
    indices[name] = vals

consensuses, mat = buildMarkerMatrix(consensuses, indices)
eqtlEnrich(consensuses, mat, tissues, eqtlDf, highSpec, eqtlPos, "ENCODE")
'''
