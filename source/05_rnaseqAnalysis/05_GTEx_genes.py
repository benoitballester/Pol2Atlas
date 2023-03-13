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
df = pd.read_csv(paths.gtexGeneCounts, sep="\t", skiprows=2).T
df 
# %%
counts = df.iloc[2:].values.astype("int32")
order = np.array(df.index[2:])
# %%
annotation = pd.read_csv(paths.gtexData + "/tsvs/sample_annot.tsv", 
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
acc = matrix_utils.looKnnCV(feat, annFull, "correlation", 5)
# %%
# Plot UMAP of samples for visualization
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(feat)
import plotly.express as px
dfPlot = pd.DataFrame(embedding, columns=["x","y"])
dfPlot["Organ"] = annotation.loc[order]["tissue_type"].values
dfPlot["Organ detailled"] = annotation.loc[order]["tissue_type_detail"].values
dfPlot = dfPlot.sample(frac=1)
colormap = dict(zip(colors.index, colors["color_hex"])) 
fig = px.scatter(dfPlot, x="x", y="y", color="Organ detailled", color_discrete_map=colormap,
                hover_data=['Organ detailled'], width=1200, height=800, title=f"54 Tissue KNN classification balanced accuracy : {acc}")
fig.update_traces(marker=dict(size=3*np.sqrt(len(dfPlot)/7500)))
fig.show()
fig.write_image(paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples_gene_centric.pdf")
fig.write_html(paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples_gene_centric.pdf" + ".html")
# %%
# Probe-gene expression correlation
from lib.pyGREATglm import pyGREAT as pyGREAT
enricher = pyGREAT(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
# %%
# Load Pol II consensuses
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensuses.columns = ["Chromosome", "Start", "End", "Name", "Score", "Strand", "ThickStart", "ThickEnd"]
# %%
# Find inferred gene relationship
import pyranges as pr
geneReg = pr.PyRanges(enricher.geneRegulatory)
consensuses = pr.PyRanges(consensuses)
linkedGene = consensuses.join(geneReg).as_df()
# %%
# Load Pol II counts
countDir = paths.countsGTEx
annotationP2 = pd.read_csv(paths.gtexData + "/tsvs/sample_annot.tsv", 
                        sep="\t", index_col="specimen_id")
colors = pd.read_csv(paths.gtexData + "colors.txt", 
                        sep="\t", index_col="tissue_site_detail")
dlFiles = os.listdir(countDir + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
countsInterg = []
allReads = []
order2 = []
allStatus = []
for f in dlFiles:
    try:
        id = ".".join(f.split(".")[:-3])
        # countsBG.append(pd.read_csv(paths.countDirectory + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                                header=None, index_col=0, sep="\t", skiprows=1).T
        countsInterg.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values.astype("int32"))
        allStatus.append(status)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order2.append(f.split(".")[0])
    except:
        print(f, "missing")
        continue
allReads = np.array(allReads)
countsInterg = np.concatenate(countsInterg, axis=1).T
conv = pd.read_csv(paths.tissueToSimplified, sep="\t", index_col="Tissue")
annotationP2.loc[order2, "tissue_type_orig"] = annotationP2.loc[order2, "tissue_type"].copy()
annotationP2.loc[order2, "tissue_type"] = conv.loc[annotationP2.loc[order2]["tissue_type"].values]["Simplified"].values
annTxt2 = annotationP2.loc[order2]["tissue_type"]
ann2, eq2 = pd.factorize(annTxt2)

# %%
sfGenes = pd.Series(sf, index=df.index[2:][np.isin(df.index[2:], order)])
sfGenes = sfGenes.loc[order2]
normed = countsInterg / sfGenes.values[:, None]
genes = df.loc["Description"][nzCounts].values

normedGenes = pd.DataFrame(counts/sfGenes.values[:, None], index=df.index[2:][np.isin(df.index[2:], order)], columns=genes)
normedGenes = normedGenes.loc[order2]
# %%
# Compute all spearman correlations of expression between inferred gene and Pol II probe
from scipy.stats import spearmanr
means = []
corr_name = []
correlations = []
corrP = []
for i in range(len(linkedGene)):
    gene = linkedGene.iloc[i]["gene_name"]
    p2 = linkedGene.iloc[i]["Name"] 
    try:
        exprGene = normedGenes[gene].values
        if len(exprGene.shape) > 1:
            print(exprGene.shape, gene)
            continue
    except KeyError:
        print("Missing", gene)
        continue
    exprP2 = normed[:, p2]
    corr_name.append((p2, gene))
    r, p = spearmanr(exprP2, exprGene)
    correlations.append(r)
    corrP.append(p)
    means.append((np.mean(exprP2),np.mean(exprGene)))
means = np.array(means)
# %%
plt.scatter(np.log(means[:, 0]), correlations, s=0.3, linewidths=0)
# %%
plt.scatter(np.log(means[:, 1]), -np.log10(corrP), s=0.3, linewidths=0)
# %%
plt.scatter(np.log(means[:, 0]), -np.log10(corrP), s=0.3, linewidths=0)
# %%
plt.scatter(np.log(means[:, 0]), np.log(means[:, 1]), s=1, linewidths=0)
# %%
gene = "FOXA1"
p2 = 49151
exprGene = normedGenes[gene].values
exprP2 = normed[:, p2]
r, p = spearmanr(exprP2, exprGene)
print(r, p)
plt.scatter(np.log(exprGene), np.log(exprP2), s=0.2)
# %%
# %%
gene = "FOXA1"
p2 = 49148
exprGene = normedGenes[gene].values
exprP2 = normed[:, p2]
r, p = spearmanr(exprP2, exprGene)
print(r, p)
plt.scatter(np.log(exprGene), np.log(exprP2), s=0.2)
# %%
palette, colors = plot_utils.getPalette(ann2)
gene = "FOXA1"
p2 = 49148
exprGene = normedGenes[gene].values
exprP2 = normed[:, p2]
r, p = spearmanr(exprP2, exprGene)
print(r, p)
plt.figure(dpi=300, figsize=(3,3))
plt.scatter(np.log10(exprGene+1), np.log10(exprP2+1), s=0.5, linewidths=0, c=colors)
plt.xlabel(f"{gene} expression")
plt.ylabel(f"Probe #{p2} expression")
plt.title(f"Spearman r : {int(r*1000)/1000.0}\np : {int(p*1000)/1000.0}")
# %%
