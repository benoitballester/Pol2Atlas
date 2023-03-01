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
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scran = importr("scran")
# %%
annotation = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
dlFiles = os.listdir(paths.countsTCGA + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
for f in np.array(dlFiles):
    try:
        id = f.split(".")[0]
        # countsBG.append(pd.read_csv(paths.countsTCGA + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(paths.countsTCGA + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(paths.countsTCGA + "500centroid/" + f, header=None, skiprows=2).values.astype("int32"))
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except:
        continue
allReads = np.array(allReads)
counts = np.concatenate(counts, axis=1).T
# bgCounts = np.concatenate(countsBG, axis=1).T

# %%
# Keep tumoral samples
kept = np.isin(order, annotation.index)
counts = counts[kept]
# bgCounts = bgCounts[:, kept]
allReads = allReads[kept]
annotation = annotation.loc[np.array(order)[kept]]
tumor = np.logical_not(annotation["Sample Type"] == "Solid Tissue Normal")
# %%
try:
    os.mkdir(paths.outputDir + "rnaseq/count_tables/")
except:
    pass
try:
    os.mkdir(paths.outputDir + "rnaseq/count_tables/TCGA/")
except:
    pass
rnaseqFuncs.saveDataset(counts, pd.DataFrame(order), paths.outputDir + "rnaseq/count_tables/TCGA/")
annotation.to_csv(paths.outputDir + "rnaseq/count_tables/TCGA/annotation_table.csv")
# %%
# Remove undected Pol II probes
nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
counts = counts[:, nzCounts]
# %%
# Normalize
sf = rnaseqFuncs.scranNorm(counts)
# %%
try:
    os.mkdir(paths.outputDir + "rnaseq/TCGA2/")
except FileExistsError:
    pass
# %%
# Feature selection 
countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf)
hv = countModel.hv
# %%
# Identify DE per condition
try:
    os.mkdir(paths.outputDir + "rnaseq/TCGA2/DE/")
except FileExistsError:
    pass
# %%
from lib.pyGREATglm import pyGREAT as pyGREATglm
enricherglm = pyGREATglm(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensuses.columns = ["Chromosome", "Start", "End", "Name", "Score", "Strand", "ThickStart", "ThickEnd"]
# %%
# 1 vs All DE analysis (markers)
from lib.utils.reusableUtest import mannWhitneyAsymp
import scipy.stats as ss
# tester = mannWhitneyAsymp(countModel.normed)
pctThreshold = 0.1
lfcMin = 0.25
orig = annotation["Project ID"]
state = np.array(["Normal"]*len(orig))
state[tumor] = "Tumor"
annotConv = pd.read_csv(paths.tcgaToMainAnnot, sep="\t", index_col=0)
conv = pd.read_csv(paths.tissueToSimplified, sep="\t", index_col="Tissue")
orig = annotConv.loc[orig]["Origin"]
orig = conv.loc[orig]["Simplified"]
orig = pd.Series([i.replace("/", "-") for i in orig.values], index=orig.index)
label = orig.str.cat(state,sep="_")

for i in label.unique():
    print(i)
    if not type(i) == str:
        continue
    labels = (label == i).astype(int)
    res2 = ss.ttest_ind(countModel.residuals[label == i], countModel.residuals[label != i], axis=0,
                        alternative="greater")
    sig = fdrcorrection(res2[1])[0]
    minpct = np.mean(counts[label == i] > 0.5, axis=0) > max(pctThreshold, 1.5/labels.sum())
    fc = np.mean(countModel.normed[label == i], axis=0) / (1e-9+np.mean(countModel.normed[label != i], axis=0))
    lfc = np.log2(fc) > lfcMin
    print(sig.sum())
    sig = sig & lfc & minpct
    print(sig.sum())
    res = pd.DataFrame(res2[::-1], columns=consensuses.index[nzCounts], index=["pval", "stat"]).T
    res["Upreg"] = sig.astype(int)
    res.to_csv(paths.outputDir + f"rnaseq/TCGA2/DE/res_{i}.csv")
    test = consensuses[nzCounts][sig]
    test.to_csv(paths.outputDir + f"rnaseq/TCGA2/DE/bed_{i}", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
    """     
    pvals = enricherglm.findEnriched(test, background=consensuses)
    enricherglm.plotEnrichs(pvals)
    enricherglm.clusterTreemap(pvals, score="-log10(pval)", 
                                output=paths.outputDir + f"rnaseq/TCGA2/DE/great_{i}.pdf") """

# %%
# Compute PCA
feat = countModel.residuals[:, hv]
decomp, model = rnaseqFuncs.permutationPA_PCA(feat, max_rank=2000, returnModel=True)
# %%
# UMAP
import umap
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(decomp)
# %%
# Plot UMAP
import plotly.express as px
import plotly.graph_objects as go
df = pd.DataFrame(embedding, columns=["x","y"])
df["Sample Type"] = annotation["Sample Type"].values
df["Project"] = annotation["Project ID"].values
df["Normal"] = 1.0*(annotation["Sample Type"].values == "Solid Tissue Normal")
df["Size"] = 3*np.sqrt(len(df)/7500) * (1+df["Normal"])
df["Hover"] = df["Project"] + "\n" + df["Sample Type"]
project_id = annotation["project_id"]
cancerType, eq = pd.factorize(annotation["project_id"], sort=True)
palette, colors = getPalette(cancerType)
df["Color"] = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
markers = go.scattergl.Marker(color=df["Color"], size=df["Size"], 
                            line=dict(width=df["Normal"], color="rgb(0,0,0)"))
dat = go.Scattergl(x=embedding[:,0],y=embedding[:,1], mode="markers",
                   marker=markers, hovertext=df["Hover"])
layout = dict(height=1200, width=1200)
fig = go.Figure(dat, layout=layout)
fig.show()
fig.write_image(paths.outputDir + "rnaseq/TCGA2/umap_samples.pdf")
fig.write_html(paths.outputDir + "rnaseq/TCGA2/umap_samples.pdf" + ".html")
# %%
import plotly.express as px
#%%
# Predictive model
import catboost
from sklearn.model_selection import train_test_split, StratifiedKFold
project_id = annotation["project_id"]
# cancerType, eq = pd.factorize(tcgaProjects["Origin"].loc[project_id])
cancerType, eq = pd.factorize(annotation["project_id"], sort=True)
pred = np.zeros(len(cancerType), dtype=int)
for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(decomp, cancerType):
    # Fit power transform on train data only
    x_train = decomp[train]
    # Fit classifier on scaled train data
    model = catboost.CatBoostClassifier(class_weights=len(cancerType) / (len(np.unique(cancerType)) * np.bincount(cancerType)), random_state=42)
    model.fit(x_train, cancerType[train], silent=True)
    # Scale and predict on test data
    x_test = decomp[test]
    pred[test] = model.predict(x_test).ravel()
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,balanced_accuracy_score
acc = balanced_accuracy_score(cancerType, pred)
# %%
# Legend of UMAP + balanced accuracy
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch

plt.figure(dpi=500)
palette, colors = getPalette(cancerType)
# allReadsScaled = (allReads - allReads.min()) / (allReads.max()-allReads.min())
# col = sns.color_palette("rocket_r", as_cmap=True)(allReadsScaled)
scale = min(10.0,100/np.sqrt(len(embedding)))
plt.scatter(embedding[:, 0], embedding[:, 1], s=scale*(3.0-2.0*tumor),
            linewidths=(1.0-tumor)*0.5, edgecolors=(0.1,0.1,0.1),c=colors)
xScale = plt.xlim()[1] - plt.xlim()[0]
yScale = plt.ylim()[1] - plt.ylim()[0]
# plt.gca().set_aspect(xScale/yScale)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/TCGA2/umap_all_tumors_plus_normal.png")
plt.show()
plt.figure(dpi=500)
plt.axis('off')
patches = []
for i in np.unique(cancerType):
    legend = Patch(color=palette[i], label=eq[i][5:])
    patches.append(legend)
plt.legend(handles=patches)
plt.title(f"Balanced accuracy on {len(np.unique(cancerType))} cancers : {acc}")
plt.savefig(paths.outputDir + "rnaseq/TCGA2/umap_all_tumors_lgd.pdf", bbox_inches="tight")
plt.show()
# %%
# Subtype info
subtypeAnnot = pd.read_csv(paths.tcgaSubtypes, sep=",", index_col="pan.samplesID")
subtypeAnnot.index = ["-".join(i.split("-")[:4]) for i in subtypeAnnot.index]
query = annotation["Sample ID"]
inSubtypeAnnot = np.isin(query, subtypeAnnot.index)
subtype = pd.Series(["N/A"]*len(query), index=query)
subtype[inSubtypeAnnot] = subtypeAnnot.loc[query[inSubtypeAnnot]]["Subtype_Selected"]
# %%
# Plot UMAP w/ subtype info
import plotly.express as px
import plotly.graph_objects as go
df = pd.DataFrame(embedding, columns=["x","y"])
palette = np.array(sns.color_palette("Paired"))[[3,9,4,5,1, -2]]
df["Sample Type"] = annotation["Sample Type"].values
df["Project"] = annotation["Project ID"].values
df["Normal"] = 1.0*(annotation["Sample Type"].values == "Solid Tissue Normal")
df["Size"] = 3*np.sqrt(len(df)/7500) * (1+df["Normal"])
df["Hover"] = df["Project"] + "\n" + df["Sample Type"]  + "\n" + subtype.values
project_id = annotation["project_id"]
brcaOnly = subtype.copy()
brcaOnly[["BRCA" not in i for i in subtype.values]] = "N/A"
subtypecat, eq = pd.factorize(brcaOnly.values, sort=True)
colors = palette[subtypecat]
nas = brcaOnly.values == "N/A"
colors = np.concatenate([colors, np.ones((len(colors),1))], axis=1)
colors[nas] = (0.5,0.5,0.5,0.1)
palette[eq=="N/A"] = (0.5,0.5,0.5)
df["Color"] = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
#df = df[df["Project"] == "TCGA-BRCA"]
markers = go.scattergl.Marker(color=df["Color"], size=df["Size"], 
                            line=dict(width=df["Normal"], color="rgb(0,0,0)"))
dat = go.Scattergl(x=df["x"],y=df["y"], mode="markers",
                   marker=markers, hovertext=df["Hover"])
layout = dict(height=1200, width=1200)
fig = go.Figure(dat, layout=layout)
fig.show()
fig.write_image(paths.outputDir + "rnaseq/TCGA2/umap_samples_subtype_BRCA.pdf")
fig.write_html(paths.outputDir + "rnaseq/TCGA2/umap_samples_subtype_BRCA.pdf" + ".html")
plt.figure(dpi=500)
plt.axis('off')
patches = []
for i in np.unique(subtypecat):
    legend = Patch(color=palette[i], label=eq[i])
    patches.append(legend)
plt.legend(handles=patches)
plt.savefig(paths.outputDir + "rnaseq/TCGA2/umap_subtype_brca_lgd.pdf", bbox_inches="tight")
plt.show()
# %%
import plotly.express as px
import plotly.graph_objects as go
df = pd.DataFrame(embedding, columns=["x","y"])
palette = np.array(sns.color_palette())
df["Sample Type"] = annotation["Sample Type"].values
df["Project"] = annotation["Project ID"].values
df["Normal"] = 1.0*(annotation["Sample Type"].values == "Solid Tissue Normal")
df["Size"] = 3*np.sqrt(len(df)/7500) * (1+df["Normal"])
df["Hover"] = df["Project"] + "\n" + df["Sample Type"]  + "\n" + subtype.values
project_id = annotation["project_id"]
brcaOnly = subtype.copy()
brcaOnly[["THCA" not in i for i in subtype.values]] = "N/A"
subtypecat, eq = pd.factorize(brcaOnly.values, sort=True)
colors = palette[subtypecat]
nas = brcaOnly.values == "N/A"
colors = np.concatenate([colors, np.ones((len(colors),1))], axis=1)
colors[nas] = (0.5,0.5,0.5,0.1)
palette[np.argmax(eq=="N/A")] = (0,0,0)
df["Color"] = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
#df = df[df["Project"] == "TCGA-BRCA"]
markers = go.scattergl.Marker(color=df["Color"], size=df["Size"], 
                            line=dict(width=df["Normal"], color="rgb(0,0,0)"))
dat = go.Scattergl(x=df["x"],y=df["y"], mode="markers",
                   marker=markers, hovertext=df["Hover"])
layout = dict(height=1200, width=1200)
fig = go.Figure(dat, layout=layout)
fig.show()
fig.write_image(paths.outputDir + "rnaseq/TCGA2/umap_samples_subtype_THCA.pdf")
fig.write_html(paths.outputDir + "rnaseq/TCGA2/umap_samples_subtype_THCA.pdf" + ".html")
plt.figure(dpi=500)
plt.axis('off')
patches = []
for i in np.unique(subtypecat):
    legend = Patch(color=palette[i], label=eq[i])
    patches.append(legend)
plt.legend(handles=patches)
plt.savefig(paths.outputDir + "rnaseq/TCGA2/umap_subtype_thca_lgd.pdf", bbox_inches="tight")
plt.show()
# %%
# Grade
import plotly.express as px
import plotly.graph_objects as go
df = pd.DataFrame(embedding, columns=["x","y"])

df["Sample Type"] = annotation["Sample Type"].values
df["Project"] = annotation["Project ID"].values
df["Normal"] = 1.0*(annotation["Sample Type"].values == "Solid Tissue Normal")
df["Size"] = 3*np.sqrt(len(df)/7500) * (1+df["Normal"])
df["Stage"] = annotation["ajcc_pathologic_stage"].values
df["Hover"] = df["Project"] + "\n" + df["Sample Type"]  + "\n" + subtype.values + "\n" + annotation["ajcc_pathologic_stage"].values

project_id = annotation["project_id"]
brcaOnly = subtype.copy()
subtypecat, eq = pd.factorize(df["Stage"].astype("str"), sort=True)
palette = np.array(sns.color_palette("rocket", len(eq)))
colors = palette[subtypecat]
colors = np.concatenate([colors, np.ones((len(colors),1))], axis=1)
# palette[eq=="N/A"] = (0.5,0.5,0.5)
df["Color"] = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
df = df[df["Project"] == "TCGA-BRCA"]
markers = go.scattergl.Marker(color=df["Color"], size=df["Size"]*2, 
                            line=dict(width=df["Normal"], color="rgb(0,0,0)"))
dat = go.Scattergl(x=df["x"],y=df["y"], mode="markers",
                   marker=markers, hovertext=df["Hover"])
layout = dict(height=1200, width=1200)
fig = go.Figure(dat, layout=layout)
fig.show()
fig.write_image(paths.outputDir + "rnaseq/TCGA2/umap_samples_BRCA_grade.pdf")
fig.write_html(paths.outputDir + "rnaseq/TCGA2/umap_samples_BRCA_grade.pdf" + ".html")
# %%
# HC 
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.residuals.T, "correlation")
# Plot Heatmap
vals = countModel.normed
plot_utils.plotHC(vals.T, eq[cancerType], vals.T,  
                    rowOrder=rowOrder, colOrder=colOrder, hq=True)
plt.savefig(paths.outputDir + "rnaseq/TCGA2/HM_all.pdf", bbox_inches="tight")
# %%
# HC (hv probes only)
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.residuals.T[hv], "correlation")
# Plot Heatmap
vals = countModel.normed
plot_utils.plotHC(vals.T[hv], eq[cancerType], vals.T[hv],  
                    rowOrder=rowOrder, colOrder=colOrder, hq=True)
plt.savefig(paths.outputDir + "rnaseq/TCGA2/HM_hv.pdf", bbox_inches="tight")

# %%
