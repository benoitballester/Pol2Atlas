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

countDir = paths.countsENCODE

# %%
annotation = pd.read_csv(paths.encodeAnnot, 
                        sep="\t", index_col=0)
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
        counts.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values.astype("int32"))
        allStatus.append(status)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(f.split(".")[0])
    except:
        print(f, "missing")
        continue
allReads = np.array(allReads)
counts = np.concatenate(counts, axis=1).T
ann, eq = pd.factorize(annotation.loc[order]["Annotation"])
# %%
nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=2)
counts = counts[:, nzCounts]
sf = rnaseqFuncs.scranNorm(counts)
countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf, maxThreads=32)
hv = countModel.hv
# %%
# Non tailed only
try:
    os.mkdir(paths.outputDir + "dist_to_genes/encode_notail/")
except FileExistsError:
    pass
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
subsets = ["5000", "7000", "9000"]
for i in range(3):
    with open(paths.outputDir + f"dist_to_genes/encode_notail/stats_{subsets[i]}.txt", "w") as f:
        print("----" + subsets[i], file=f)
        tailed = pd.read_csv(paths.outputDir + f"dist_to_genes/pol2_{subsets[i]}_TES_ext.bed", sep="\t")
        nonTailed = np.isin(np.arange(len(consensuses))[nzCounts], tailed["Name"].values)
        nonTailed = np.logical_not(nonTailed)
        print("Kept ", nonTailed.sum(), "probes", file=f)
        avgExpr = np.log(np.mean(countModel.normed, axis=0))
        cat = np.array(["Tail of gene"]*len(avgExpr))
        cat[nonTailed] = "Non tail of gene"
        avgExpr = pd.DataFrame({"Log Average expression":avgExpr, "Category":cat})
        plt.figure(dpi=500)
        sns.boxplot(data=avgExpr, x="Log Average expression", y="Category", showfliers=False)
        sns.stripplot(data=avgExpr, x="Log Average expression", y="Category", dodge=True, 
                            jitter=0.4, s=0.5, alpha=1.0)
        plt.savefig(paths.outputDir + f"dist_to_genes/encode_notail/counts_{subsets[i]}.pdf", bbox_inches="tight")
        feat = countModel.residuals[:, nonTailed][:, hv[nonTailed]]
        decomp, model = rnaseqFuncs.permutationPA_PCA(feat, 1, max_rank=509, returnModel=True)
        print("Leave-one-out KNN weighted accuracy : ", matrix_utils.looKnnCV(decomp, ann, "correlation", 1), file=f)
        # Plot UMAP of samples for visualization
        embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                            metric="correlation").fit_transform(decomp)
        import plotly.express as px
        df = pd.DataFrame(embedding, columns=["x","y"])
        df["Annotation"] = annotation.loc[order]["Annotation"].values
        df["Biosample term name"] = annotation.loc[order]["Biosample term name"].values

        annot, palette, colors = plot_utils.applyPalette(annotation.loc[order]["Annotation"],
                                                        np.unique(annotation.loc[order]["Annotation"]),
                                                        paths.polIIannotationPalette, ret_labels=True)
        palettePlotly = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in palette]
        colormap = dict(zip(annot, palettePlotly))     

        fig = px.scatter(df, x="x", y="y", color="Annotation", color_discrete_map=colormap,
                        hover_data=['Biosample term name'], width=800, height=800)
        fig.show()
        fig.write_image(paths.outputDir + f"dist_to_genes/encode_notail/umap_samples_{subsets[i]}.pdf")
        fig.write_html(paths.outputDir + f"dist_to_genes/encode_notail/umap_samples{subsets[i]}.pdf" + ".html")
# %%
# Tailed only
try:
    os.mkdir(paths.outputDir + "dist_to_genes/encode_tail/")
except FileExistsError:
    pass
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
subsets = ["5000", "7000", "9000"]
for i in range(3):
    with open(paths.outputDir + f"dist_to_genes/encode_tail/stats_{subsets[i]}.txt", "w") as f:
        print("----" + subsets[i], file=f)
        tailed = pd.read_csv(paths.outputDir + f"dist_to_genes/pol2_{subsets[i]}_TES_ext.bed", sep="\t")
        nonTailed = np.isin(np.arange(len(consensuses))[nzCounts], tailed["Name"].values)
        print("Kept ", nonTailed.sum(), "probes", file=f)
        avgExpr = np.log(np.mean(countModel.normed, axis=0))
        cat = np.array(["Tail of gene"]*len(avgExpr))
        cat[nonTailed] = "Non tail of gene"
        avgExpr = pd.DataFrame({"Log Average expression":avgExpr, "Category":cat})
        plt.figure(dpi=500)
        sns.boxplot(data=avgExpr, x="Log Average expression", y="Category", showfliers=False)
        sns.stripplot(data=avgExpr, x="Log Average expression", y="Category", dodge=True, 
                            jitter=0.4, s=0.5, alpha=1.0)
        plt.savefig(paths.outputDir + f"dist_to_genes/encode_tail/counts_{subsets[i]}.pdf", bbox_inches="tight")
        feat = countModel.residuals[:, nonTailed][:, hv[nonTailed]]
        decomp, model = rnaseqFuncs.permutationPA_PCA(feat, 1, max_rank=509, returnModel=True)
        print("Leave-one-out KNN weighted accuracy : ", matrix_utils.looKnnCV(decomp, ann, "correlation", 1), file=f)
        # Plot UMAP of samples for visualization
        embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                            metric="correlation").fit_transform(decomp)
        import plotly.express as px
        df = pd.DataFrame(embedding, columns=["x","y"])
        df["Annotation"] = annotation.loc[order]["Annotation"].values
        df["Biosample term name"] = annotation.loc[order]["Biosample term name"].values

        annot, palette, colors = plot_utils.applyPalette(annotation.loc[order]["Annotation"],
                                                        np.unique(annotation.loc[order]["Annotation"]),
                                                        paths.polIIannotationPalette, ret_labels=True)
        palettePlotly = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in palette]
        colormap = dict(zip(annot, palettePlotly))     

        fig = px.scatter(df, x="x", y="y", color="Annotation", color_discrete_map=colormap,
                        hover_data=['Biosample term name'], width=800, height=800)
        fig.show()
        fig.write_image(paths.outputDir + f"dist_to_genes/encode_tail/umap_samples_{subsets[i]}.pdf")
        fig.write_html(paths.outputDir + f"dist_to_genes/encode_tail/umap_samples{subsets[i]}.pdf" + ".html")

# %%
